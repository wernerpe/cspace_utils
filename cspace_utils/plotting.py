from pydrake.all import (Hyperellipsoid, 
                         HPolyhedron,
                         MathematicalProgram,
                         VPolytope, 
                         Solve,
                         Rgba,
                         SurfaceTriangle,
                         TriangleSurfaceMesh,
                         Cylinder,
                         RotationMatrix,
                         RigidTransform,
                         Sphere)
from scipy.spatial import ConvexHull
from scipy.linalg import block_diag
from cspace_utils.geometry import (arrange_homogeneous_ellipse_matrix_to_vector,  
                                   build_quadratic_features,
                                   stretch_array_to_3d,
                                   get_AABB_limits,
                                   get_AABB_limits_hyperelliposid)
from cspace_utils.colors import (generate_maximally_different_colors)
import numpy as np
from functools import partial
import mcubes

def sorted_vertices(vpoly):
    assert vpoly.ambient_dimension() == 2
    poly_center = np.sum(vpoly.vertices(), axis=1) / vpoly.vertices().shape[1]
    vertex_vectors = vpoly.vertices() - np.expand_dims(poly_center, 1)
    sorted_index = np.arctan2(vertex_vectors[1], vertex_vectors[0]).argsort()
    return vpoly.vertices()[:, sorted_index]


def plot_HPoly(ax, HPoly, color = None, zorder = 0):
    v = sorted_vertices(VPolytope(HPoly)).T#s
    v = np.concatenate((v, v[0,:].reshape(1,-1)), axis=0)
    if color is None:
        p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7, zorder = zorder)
    else:
        p = ax.plot(v[:,0], v[:,1], linewidth = 2, alpha = 0.7, c = color, zorder = zorder)

    ax.fill(v[:,0], v[:,1], alpha = 0.5, c = p[0].get_color(), zorder = zorder)



def plot_ellipse(ax, H, n_samples, color = None, linewidth = 1):
    A = H.A()
    center = H.center()
    angs = np.linspace(0, 2*np.pi, n_samples+1)
    coords = np.zeros((2, n_samples + 1))
    coords[0, :] = np.cos(angs)
    coords[1, :] = np.sin(angs)
    Bu = np.linalg.inv(A)@coords
    pts = Bu + center.reshape(2,1)
    if color is None:
        ax.plot(pts[0, :], pts[1, :], linewidth = linewidth)
    else:
        ax.plot(pts[0, :], pts[1, :], linewidth = linewidth, color = color)

def plot_ellipse_homogenous_rep(ax, Emat, xrange, yrange, resolution, linewidth =1, color = 'b', zorder = 1,):
    # center = np.linalg.solve(-Emat[:-1, :-1], Emat[-1, :-1])
    # max = np.max(1/(np.sqrt(np.linalg.eigh(Emat[:-1, :-1])[0] + 1e-3)))
    # if clip is None:
    #     clip = [-10,10]
    x = np.arange(xrange[0], xrange[1], resolution)
    y = np.arange(yrange[0], yrange[1], resolution)
    X,Y = np.meshgrid(x,y)
    hE = arrange_homogeneous_ellipse_matrix_to_vector(Emat)
    Z = (hE@build_quadratic_features(np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)), axis = 1)).T).reshape(len(y), len(x))
    CS = ax.contour(X, Y, Z,[1.0], linewidths = linewidth,zorder = zorder, colors = [color])
    return CS


def plot_surface(meshcat_instance,
                 path,
                 X,
                 Y,
                 Z,
                 rgba=Rgba(.87, .6, .6, 1.0),
                 wireframe=False,
                 wireframe_line_width=1.0):
    # taken from
    # https://github.com/RussTedrake/manipulation/blob/346038d7fb3b18d439a88be6ed731c6bf19b43de/manipulation/meshcat_cpp_utils.py#L415
    (rows, cols) = Z.shape
    assert (np.array_equal(X.shape, Y.shape))
    assert (np.array_equal(X.shape, Z.shape))

    vertices = np.empty((rows * cols, 3), dtype=np.float32)
    vertices[:, 0] = X.reshape((-1))
    vertices[:, 1] = Y.reshape((-1))
    vertices[:, 2] = Z.reshape((-1))

    # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy  # noqa
    faces = np.empty((rows - 1, cols - 1, 2, 3), dtype=np.uint32)
    r = np.arange(rows * cols).reshape(rows, cols)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces.shape = (-1, 3)

    meshcat_instance.SetTriangleMesh(
        path,
        vertices.T,
        faces.T,
        rgba,
        wireframe,
        wireframe_line_width)


def plot_point(point, meshcat_instance, name,
               color=Rgba(0.06, 0.0, 0, 1), radius=0.01):
    meshcat_instance.SetObject(name,
                               Sphere(radius),
                               color)
    meshcat_instance.SetTransform(name, RigidTransform(
        RotationMatrix(), stretch_array_to_3d(point)))

def plot_points(meshcat, points, name, size = 0.05, color = Rgba(0.06, 0.0, 0, 1)):
    for i, pt in enumerate(points):
        n_i = name+f"/pt{i}"
        plot_point(pt, meshcat, n_i, color = color, radius=size)
        

def plot_polytope(polytope, meshcat_instance, name,
                  resolution=50, color=None,
                  wireframe=True,
                  random_color_opacity=0.2,
                  fill=True,
                  line_width=10):
    if color is None:
        color = Rgba(*np.random.rand(3), random_color_opacity)
    if polytope.ambient_dimension == 3:
        verts, triangles = get_plot_poly_mesh(polytope,
                                              resolution=resolution)
        meshcat_instance.SetObject(name, TriangleSurfaceMesh(triangles, verts),
                                   color, wireframe=wireframe)

    else:
        plot_hpoly2d(polytope, meshcat_instance, name,
                     color,
                     line_width=line_width,
                     fill=fill,
                     resolution=resolution,
                     wireframe=wireframe)



def plot_hpoly2d(polytope, meshcat_instance, name,
                 color,
                 line_width=8,
                 fill=False,
                 resolution=30,
                 wireframe=True):
    # plot boundary
    vpoly = VPolytope(polytope)
    verts = vpoly.vertices()
    hull = ConvexHull(verts.T)
    inds = np.append(hull.vertices, hull.vertices[0])
    hull_drake = verts.T[inds, :].T
    hull_drake3d = np.vstack([hull_drake, np.zeros(hull_drake.shape[1])])
    color_RGB = Rgba(color.r(), color.g(), color.b(), 1)
    meshcat_instance.SetLine(name, hull_drake3d,
                             line_width=line_width, rgba=color_RGB)
    if fill:
        width = 0.5
        C = block_diag(polytope.A(), np.array([-1, 1])[:, np.newaxis])
        d = np.append(polytope.b(), width * np.ones(2))
        hpoly_3d = HPolyhedron(C, d)
        verts, triangles = get_plot_poly_mesh(hpoly_3d,
                                              resolution=resolution)
        meshcat_instance.SetObject(name + "/fill",
                                   TriangleSurfaceMesh(triangles, verts),
                                   color, wireframe=wireframe)


def get_plot_poly_mesh(polytope, resolution):
    def inpolycheck(q0, q1, q2, A, b):
        q = np.array([q0, q1, q2])
        res = np.min(1.0 * (A @ q - b <= 0))
        return res

    aabb_max, aabb_min = get_AABB_limits(polytope)

    col_hand = partial(inpolycheck, A=polytope.A(), b=polytope.b())
    vertices, triangles = mcubes.marching_cubes_func(tuple(aabb_min),
                                                     tuple(aabb_max),
                                                     resolution,
                                                     resolution,
                                                     resolution,
                                                     col_hand,
                                                     0.5)
    tri_drake = [SurfaceTriangle(*t) for t in triangles]
    return vertices, tri_drake


def draw_traj(meshcat_instance, traj, maxit, name = "/trajectory",
              color = Rgba(0,0,0,1), line_width = 3):
    pts = np.squeeze(np.array([traj.value(it * traj.end_time() / maxit) for it in range(maxit)]))
    pts_3d = np.hstack([pts, 0 * np.ones((pts.shape[0], 3 - pts.shape[1]))]).T
    meshcat_instance.SetLine(name, pts_3d, line_width, color)


def plot_regions(meshcat, regions, ellipses = None,
                     region_suffix = '', colors = None,
                     wireframe = False,
                     opacity = 0.7,
                     fill = True,
                     line_width = 10,
                     darken_factor = .2,
                     el_opacity = 0.3,
                     resolution = 30,
                     offset = np.zeros(3)):
        if colors is None:
            colors = generate_maximally_different_colors(len(regions))

        for i, region in enumerate(regions):
            c = Rgba(*[col for col in colors[i]],opacity)
            prefix = f"/iris/regions{region_suffix}/{i}"
            name = prefix + "/hpoly"
            if region.ambient_dimension() == 3:
                # plot_hpoly3d(meshcat, name, region,
                #                   c, wireframe = wireframe, resolution = resolution, offset = offset)
                plot_hpoly3d_2(meshcat, name, region,
                                  c, wireframe = wireframe, resolution = resolution, offset = offset)

def plot_hpoly3d(meshcat, name, hpoly, color, wireframe = True, resolution = 30, offset = np.zeros(3)):
        verts, triangles = get_plot_poly_mesh(hpoly,
                                                   resolution=resolution)
        meshcat.SetObject(name, TriangleSurfaceMesh(triangles, verts+offset.reshape(-1,3)),
                                color, wireframe=wireframe)
        
def plot_hpoly3d_2(meshcat, name, hpoly, color, wireframe = True, resolution = -1, offset = np.zeros(3)):
        #meshcat wierdness of double rendering
        hpoly = HPolyhedron(hpoly.A(), hpoly.b() + 0.05*(np.random.rand(hpoly.b().shape[0])-0.5))
        verts = VPolytope(hpoly).vertices().T
        hull = ConvexHull(verts)
        triangles = []
        for s in hull.simplices:
            triangles.append(s)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        # obj = self[name]
        # objwf = self[name+'wf']
        # col = to_hex(color)
        #material = MeshLambertMaterial(color=col, opacity=opacity)
        color2 = Rgba(0.8*color.r(), 0.8*color.g(), 0.8*color.b(), color.a())
        meshcat.SetObject(name, TriangleSurfaceMesh(tri_drake, verts+offset.reshape(-1,3)),
                                color, wireframe=False)
        meshcat.SetObject(name+'wf', TriangleSurfaceMesh(tri_drake, verts+offset.reshape(-1,3)),
                                color2, wireframe=True)
        # #obj.set_object(TriangularMeshGeometry(verts, triangles), material)
        # material = MeshLambertMaterial(color=col, opacity=0.95, wireframe=True)
        # objwf.set_object(TriangularMeshGeometry(verts, triangles), material)

def plot_ellipses(meshcat, ellipses, name, colors, offset = None):
    for i, e in enumerate(ellipses):
        c = colors[i]
        prefix = f"/{name}/ellipses/{i}"
        plot_ellipse(meshcat, prefix, e, c, offset)

def plot_ellipse( meshcat, name, ellipse, color, offset = None):
    
        shape, pose = ellipse.ToShapeWithPose()
        if offset is not None:
            pose2 = RigidTransform(pose.rotation(), pose.translation() + offset) 
        meshcat.SetObject(name, shape, color)
        meshcat.SetTransform(name, pose2)


def plot_triad(pose, meshcat, name='triad',size = 0.2):
    h = size
    if 'targ' in name:
        colors = [Rgba(1,0.5,0, 0.5), Rgba(0.5,1,0, 0.5), Rgba(0.0,0.5,1, 0.5)]
    else:
        colors = [Rgba(1,0,0, 1), Rgba(0.,1,0, 1), Rgba(0.0,0.0,1, 1)]

    rot = pose.rotation()@RotationMatrix.MakeYRotation(np.pi/2)
    pos= pose.translation() +pose.rotation()@np.array([h/2, 0,0])
    meshcat.SetObject(f"/drake/{name}/triad1",
                                   Cylinder(size/20, size),
                                   colors[0])
    meshcat.SetTransform(f"/drake/{name}/triad1",RigidTransform(rot, pos))
    rot = pose.rotation()@RotationMatrix.MakeXRotation(-np.pi/2)
    pos= pose.translation() +pose.rotation()@np.array([0,h/2,0])

    meshcat.SetObject(f"/drake/{name}/triad2",
                                   Cylinder(size/20,size),
                                   colors[1])
    meshcat.SetTransform(f"/drake/{name}/triad2",RigidTransform(rot, pos))
    pos= pose.translation().copy()
    rot = pose.rotation()
    pos = pos + rot@np.array([0,0,h/2])
    meshcat.SetObject(f"/drake/{name}/triad3",
                                   Cylinder(size/20,size),
                                   colors[2])
    meshcat.SetTransform(f"/drake/{name}/triad3",RigidTransform(rot, pos))


import pydrake
def get_shunk_plotter(plant, scene_graph, plant_context, diagram_context):

    query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(diagram_context))
    inspector = query.inspector()
    a = inspector.GetCollisionCandidates()
    geomids= []
    for b, c in a:
        geomids.append(b)
        geomids.append(c)
    ids = list(set(geomids))
    frame_id_dict = {}
    for idx in range(len(ids)):
        #print(idx, plant.GetBodyFromFrameId(inspector.GetFrameId(ids[idx])))
        if plant.GetBodyFromFrameId(inspector.GetFrameId(ids[idx])).name() =='body':
            frame_id_dict['body'] = ids[idx]
        if plant.GetBodyFromFrameId(inspector.GetFrameId(ids[idx])).name() =='left_finger':
            frame_id_dict['left_finger'] = ids[idx]
        if plant.GetBodyFromFrameId(inspector.GetFrameId(ids[idx])).name() =='right_finger':
            frame_id_dict['right_finger'] = ids[idx]
    print(frame_id_dict)
    geom_ids = [inspector.GetGeometries(inspector.GetFrameId(frame_id_dict[k]))[0] for k in frame_id_dict.keys()]

    sh_frames = [int(plant.GetBodyByName('body').index()),int(plant.GetBodyByName('left_finger').index()),int(plant.GetBodyByName('right_finger').index())]
    sh_geom = [inspector.GetShape(id) for id in geom_ids] 
    sh_names = ['box', 'l','r']

    def plot_endeff_pose(meshcat, q, name = '', color = Rgba(1,1,0.1,0.8)):
        plant.SetPositions(plant_context, q)
        tfs = [plant.EvalBodyPoseInWorld(plant_context, plant.get_body(pydrake.multibody.tree.BodyIndex(f))) for f in sh_frames]
        for n, f, geom in zip(sh_names, tfs, sh_geom):
            meshcat.SetObject("/shunk/"+name+"/"+n,
                                    geom,
                                    color)
            meshcat.SetTransform("/shunk/"+name+"/"+n, f)
    
    def plot_endeff_poses(meshcat, qs, color = Rgba(1,1,0.1,0.8), prefix = ''):
        for i,q in enumerate(qs):
            plot_endeff_pose(meshcat, q, prefix+f"_{i}", color)
    
    return plot_endeff_poses