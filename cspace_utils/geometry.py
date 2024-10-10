import numpy as np
from pydrake.all import (Hyperellipsoid,
                         MathematicalProgram,
                         Solve, 
                         le,
                         ge,
                         eq,
                         MosekSolver,
                         VPolytope,
                         PiecewisePolynomial,
                         SolverOptions,
                         CommonSolverOption,
                         HPolyhedron)
import pydrake.symbolic as sym

def switch_ellipse_description(A, b):
    d = np.linalg.solve(A.T@A, -A.T@b)
    return Hyperellipsoid(A,d), A, d

def get_lj_ellipse_homogeneous_rep(pts):
    HE = Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(pts.T, rank_tol = 1e-10)
    return get_homogeneous_matrix(HE)

# def get_hyperellipsoid_from_homogeneous_matrix(Emat):
#     An = (np.linalg.cholesky(Emat[:-1, :-1])).T
#     center = np.linalg.solve(-Emat[:-1, :-1], Emat[-1, :-1])
#     return Hyperellipsoid(An, center)

def arrange_homogeneous_ellipse_matrix_to_vector(Emat):
    #flattens the homogenous matrix describing the ellipsoid to a vector
    # HE  = (diagonal, offdiagonal terms *2 row first ) 
    dim = Emat.shape[0]
    hE = []
    #hE = np.zeros(int(dim*(dim+1)/2))
    index = 0
    for i in range(dim):
        #hE[index] = Emat[i,i]
        hE.append(1*Emat[i,i])
        index+=1
    for i in range(dim):
        for j in range(i+1, dim):
            #hE[index] = 2*Emat[i,j]
            hE.append(2*Emat[i,j])
            index+=1
    return np.array(hE)

def build_quadratic_features(q_mat):
    #input are row-wise points in Rn
    dim = q_mat.shape[1]+1
    q_mat_hom = np.concatenate((q_mat, np.ones((q_mat.shape[0],1))), axis =1)
    num_features = int(dim*(dim+1)/2)
    features = np.zeros((q_mat.shape[0], num_features))
    index = 0
    #quadratic features
    for i in range(dim):
        features[:, index] = q_mat_hom[:, i]*q_mat_hom[:, i]
        index+=1
    #mixed features
    for i in range(dim-1):
        for j in range(i+1, dim):
            features[:, index] = q_mat_hom[:,i]*q_mat_hom[:,j]
            index+=1
    return features

def get_homogeneous_matrix(E):
    ## (q^T 1)@Emat@(q^T 1).T<=1 is the homogeneous form 
    dim = len(E.center())
    Eupp = E.A().T@E.A()
    Eoff = (-E.center().T@E.A().T@E.A()).reshape(1,-1)
    Econst = E.center().T@E.A().T@E.A()@E.center()
    Emat = np.zeros((dim+1, dim +1))
    Emat[:dim, :dim] = Eupp
    Emat[:dim, -1] = Eoff.squeeze()
    Emat[-1, :dim] = Eoff.squeeze()
    Emat[-1,-1] = Econst
    return Emat

def get_hyperellipsoid_from_homogeneous_ellipsoid_matrix(E):
    center = np.linalg.solve(E[:-1, :-1], E[-1, :-1])
    afactor = (np.linalg.cholesky(E[:-1, :-1])).T
    return Hyperellipsoid(afactor, center)

def get_AABB_limits(hpoly, dim=3):
    max_limits = []
    min_limits = []
    A = hpoly.A()
    b = hpoly.b()

    for idx in range(dim):
        aabbprog = MathematicalProgram()
        x = aabbprog.NewContinuousVariables(dim, 'x')
        cost = x[idx]
        aabbprog.AddCost(cost)
        aabbprog.AddConstraint(le(A @ x, b))
       
        result = Solve(aabbprog)
        min_limits.append(result.get_optimal_cost() - 0.01)
        aabbprog = MathematicalProgram()
        x = aabbprog.NewContinuousVariables(dim, 'x')
        cost = -x[idx]
        aabbprog.AddCost(cost)
        aabbprog.AddConstraint(le(A @ x, b))
        result = Solve(aabbprog)
        max_limits.append(-result.get_optimal_cost() + 0.01)
    return max_limits, min_limits


def get_AABB_cvxhull(regions):
    vps = [VPolytope(r).vertices().T for r in regions]
    cvxh = HPolyhedron(VPolytope(np.concatenate(tuple(vps), axis=0).T))
    max, min = get_AABB_limits(cvxh, dim = 3)    
    return np.array(min), np.array(max), cvxh

def get_AABB_limits_hyperelliposid(hell: Hyperellipsoid):
    dim = hell.ambient_dimension()

    max_limits = []
    min_limits = []
    c = hell.center()
    Q = hell.A().T@hell.A()
    Qinv = np.linalg.inv(Q)
    max_limits = np.sqrt(np.diag(Qinv))+c
    min_limits = -np.sqrt(np.diag(Qinv))+c
    return max_limits, min_limits

def stretch_array_to_3d(arr, val=0.):
    if arr.shape[0] < 3:
        arr = np.append(arr, val * np.ones((3 - arr.shape[0])))
    return arr

def generate_walk_around_polytope(h_polytope, num_verts):
    v_polytope = VPolytope(h_polytope)
    verts_to_visit_index = np.random.randint(0, v_polytope.vertices().shape[1], num_verts)
    verts_to_visit = v_polytope.vertices()[:, verts_to_visit_index]
    t_knots = np.linspace(0, 1,  verts_to_visit.shape[1])
    lin_traj = PiecewisePolynomial.FirstOrderHold(t_knots, verts_to_visit)
    return lin_traj


def ComputeHessian(poly):
    x = np.array([x for x in poly.indeterminates()])
    jac = poly.Jacobian(x)
    H = np.array([component.Jacobian(x) for component in jac])
    return H

# You can pass auxillary indets if you have free indeterminates that you want to reuse
def AddMatrixSosConstraint(prog, M, auxiliary=None):
    assert M.shape[0] == M.shape[1]
    if auxiliary and auxiliary.shape[0] < M:
        raise ValueError("Auxiliary Indeterminates too small to add this constraint")
    elif auxiliary:
        auxiliary = auxiliary[: M.shape[0]]
    else:
        auxiliary = prog.NewIndeterminates(M.shape[0], "aux")

    aux_poly = auxiliary.T @ M @ auxiliary
    Q, lam = prog.AddSosConstraint(aux_poly)
    return aux_poly, Q, lam


def AddPolynomialIsSosConvexConstraint(prog, M, auxillary_indets=None):
    H = ComputeHessian(M)
    return AddMatrixSosConstraint(prog, H, auxillary_indets)

def get_supporting_plane(pt, verts, randomize=False):
    dim = len(pt)
    prog = MathematicalProgram()
    b = prog.NewContinuousVariables(1, 'b')
    a = prog.NewContinuousVariables(dim, 'a')
    if randomize:
        rand = 2*(np.random.rand(dim)-0.5)*0.1*np.linalg.norm(pt) 
    else:
        rand = pt
    cost = - a.T@(rand) 
    prog.AddBoundingBoxConstraint( np.array(dim*[-1]),  np.array(dim*[1]), a)
    for v in verts:
        prog.AddLinearConstraint(le(a.T@v +b, 0))
    prog.AddLinearConstraint(eq(a.T@pt + b, 0))
    prog.AddLinearCost(cost)

    result= Solve(prog)
    if result.is_success():
        return result.GetSolution(a), result.GetSolution(b)
    
def get_supporting_point_direction(pt, verts, randomize =False):
    a, b = get_supporting_plane(pt, verts, randomize)
    a_norm = a/(np.linalg.norm(a)+1e-6)
    return a_norm

def get_supporting_points(verts, distances = [1], randomize=True):
    pts=[]
    distances_pts =[] 
    for pt in verts:
        for d in distances:
            dir = get_supporting_point_direction(pt, verts, randomize=randomize)
            pts.append(pt + d*dir)
            distances_pts.append(d)
    return np.array(pts), np.array(distances_pts)

def eval_poly_partial(point, p):
    (A, decision_vars, b) = p.EvaluateWithAffineCoefficients(np.array([v for v in p.indeterminates()]), point)
    return (A@decision_vars + b)[0], (A, decision_vars, b)

def polynomial_distance_function_approx(vpoly : VPolytope, 
                                        degree = 6,
                                        eval_factor = 10):
    dim = vpoly.ambient_dimension()
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(dim, 'x')
    p = prog.NewFreePolynomial(sym.Variables(x), deg=degree, coeff_name="c")
    
    #constrain distance function to be SOS-convex
    H = ComputeHessian(p)
    aux_poly, Q, lam = AddMatrixSosConstraint(prog, H)

    points_on_boundary = vpoly.GetMinimalRepresentation().vertices().T

    distances_required = int(eval_factor*len(p.decision_variables())/len(points_on_boundary)+0.5)
    supppts, distances = get_supporting_points(points_on_boundary, 
                                               distances = np.square(np.linspace(0,1.5, num= distances_required)),
                                               randomize = True)
    print(f"number of eval points {len(supppts)}")
    for val, point in zip(distances, supppts):
        expr, (A, vars, b) = eval_poly_partial(point, p)
        t = prog.NewContinuousVariables(1,"t")
        t = prog.NewContinuousVariables(1,"t")
        prog.AddLinearConstraint(le(A@vars+b-val, t[0]))
        prog.AddLinearConstraint(ge(A@vars+b-val, -t[0]))
        prog.AddLinearCost(t[0])

    solver = MosekSolver()

    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    result = solver.Solve(prog, solver_options = solver_options)
    if result.is_success():
        result_poly = result.GetSolution(p)
        def eval_poly(pt):
            return result_poly.Evaluate({v: pt[i] for i, v in enumerate(x)} )
        return result_poly, eval_poly, supppts
    else:
        print(f"Distance function approximation failed.")
        return None, None
    