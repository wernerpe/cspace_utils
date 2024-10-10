import numpy as np
from pydrake.all import (Hyperellipsoid,
                         MathematicalProgram,
                         Solve,
                         SolverOptions,
                         CommonSolverOption,
                         GurobiSolver, 
                         eq
                         ) 

from cspace_utils.geometry import (build_quadratic_features, 
                      arrange_homogeneous_ellipse_matrix_to_vector,
                      switch_ellipse_description)


def compute_outer_LJ_sphere(pts):
    dim = pts[0].shape[0]
    # pts = #[pt1, pt2]
    # for _ in range(2*dim):
    #     m = 0.5*(pt1+pt2) + eps*(np.random.rand(2,1)-0.5)
    #     pts.append(m)
    upper_triangular_indeces = []
    for i in range(dim-1):
        for j in range(i+1, dim):
            upper_triangular_indeces.append([i,j])

    upper_triangular_indeces = np.array(upper_triangular_indeces)
    prog = MathematicalProgram()
    inv_radius = prog.NewContinuousVariables(1, 'rad')
    A = inv_radius*np.eye(dim)
    b = prog.NewContinuousVariables(dim, 'b')
    prog.AddMaximizeLogDeterminantCost(A)
    for idx, pt in enumerate(pts):
        pt = pt.reshape(dim,1)
        S = prog.NewSymmetricContinuousVariables(dim+1, 'S')
        prog.AddPositiveSemidefiniteConstraint(S)
        prog.AddLinearEqualityConstraint(S[0,0] == 0.9)
        v = (A@pt + b.reshape(dim,1)).T
        c = (S[1:,1:]-np.eye(dim)).reshape(-1)
        for idx in range(dim):
            prog.AddLinearEqualityConstraint(S[0,1 + idx]-v[0,idx], 0 )
        for ci in c:
            prog.AddLinearEqualityConstraint(ci, 0 )

    prog.AddPositiveSemidefiniteConstraint(A) # eps * identity

    # for aij in A[upper_triangular_indeces[:,0], upper_triangular_indeces[:,1]]:
    #     prog.AddLinearConstraint(aij == 0)
    prog.AddPositiveSemidefiniteConstraint(10000*np.eye(dim)-A)

    sol = Solve(prog)
    if sol.is_success():
        HE, _, _ = switch_ellipse_description(sol.GetSolution(inv_radius)*np.eye(dim), sol.GetSolution(b))
    return HE

def max_clique_w_ellipsoidal_cvx_hull_constraint(adj_mat, 
                                                 graph_vertices, 
                                                 c=None, 
                                                 min_eig = 1e-3, 
                                                 max_eig = 5e-2, 
                                                 r_scale = 1.0, 
                                                 M_vals = None):
    """ 
    adj_mat: nxn {0,1} binary adjacency matrix
    graph_vertices: nxdim vertex locations
    c: nx1 cost vector for the vertices (used for computing covers)
    min_eig: minimum eigen value of decision boundary
    max_eig: maximum eigen value of decision boundary
    M_vals: nx1 setting this vector overrides the BigM values 
    """

    assert adj_mat.shape[0] == len(graph_vertices)
    assert r_scale>=0.5
    
    #assert graph_vertices[0, :].shape[0] == points_to_exclude.shape[1]
    dim = graph_vertices.shape[1]
    n = adj_mat.shape[0]
    if M_vals is None:
        #compute radius of circumscribed sphere of all points to get margin size
        HS = compute_outer_LJ_sphere(graph_vertices)
        radius = 1/(HS.A()[0,0]+1e-6)
        center = HS.center()
        dists = np.linalg.norm((graph_vertices-center.reshape(1,-1)), axis=1)
        M_vals = max_eig*(dists+r_scale*radius)**2
    else:
        assert M_vals.shape[0] ==n 

    fq = build_quadratic_features(graph_vertices)
    if c is None:
        c = np.ones((n,))
    prog = MathematicalProgram()
    v = prog.NewBinaryVariables(n)
    Emat = prog.NewSymmetricContinuousVariables(dim+1)
    hE = arrange_homogeneous_ellipse_matrix_to_vector(Emat)
    prog.AddLinearCost(-np.sum(c*v))

    for i in range(0,n):
        for j in range(i+1,n):
            if adj_mat[i,j] == 0:
                prog.AddLinearConstraint(v[i] + v[j] <= 1)

    for i in range(n):
        val = hE.T@fq[i,:]
        prog.AddLinearConstraint(val>=1-v[i])
        prog.AddLinearConstraint(val<=1+M_vals[i]*(1-v[i])) #

    # A = Emat[:-1, :-1]
    # prog.AddLinearEqualityConstraint(np.ones(dim)@A@np.ones(dim), 1)
    #force non-trivial solutions
    pd_amount = min_eig *np.eye(dim)
    prog.AddPositiveDiagonallyDominantMatrixConstraint(Emat[:-1, :-1]-pd_amount)
    #prog.AddPositiveDiagonallyDominantMatrixConstraint(max_eig_mat-Emat[:-1, :-1])
    
    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    solver_options.SetOption(GurobiSolver.id(), 'WorkLimit', 400)
    result = Solve(prog, solver_options=solver_options)
    print(result.is_success())
    return  -result.get_optimal_cost(), np.where(np.abs(result.GetSolution(v)-1)<=1e-4)[0], result.GetSolution(Emat), result.GetSolution(v), M_vals

def compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint(adj_mat, pts, smin =10, max_aspect_ratio = 50, r_scale = 1.1):
    assert adj_mat.shape[0] == len(pts)

    LJS = compute_outer_LJ_sphere(pts)
    radius = 1/(LJS.A()[0,0]+1e-6)
    min_eig = radius/10 * 1e-3
    max_eig = max_aspect_ratio*min_eig
    #radius = 1/(HS.A()[0,0]+1e-6)
    center = LJS.center()
    dists = np.linalg.norm((pts-center.reshape(1,-1)), axis=1)
    M_vals = max_eig*(dists+r_scale*radius)**2

    cliques = []
    done = False
    pts_curr = pts.copy()
    adj_curr = adj_mat.copy()
    ind_curr = np.arange(len(adj_curr))
    c = np.ones((adj_mat.shape[0],))
    boundaries = []
    while not done:
        val, ind_max_clique_local,dec_boundary,_,_ = max_clique_w_ellipsoidal_cvx_hull_constraint(adj_curr, pts_curr, c,  min_eig, max_eig, r_scale, M_vals = M_vals)
        boundaries+= [dec_boundary]
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        c[ind_max_clique_local] = 0
        cliques.append(index_max_clique_global.reshape(-1))
        if val< smin or np.sum(c) == 0:
            done = True
    return cliques, boundaries