import numpy as np
from pydrake.all import (Hyperellipsoid,
                         MathematicalProgram,
                         Solve, le,
                         VPolytope,
                         PiecewisePolynomial)

def switch_ellipse_description(A, b):
    d = np.linalg.solve(A.T@A, -A.T@b)
    return Hyperellipsoid(A,d), A, d

def get_lj_ellipse_homogeneous_rep(pts):
    HE = Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(pts.T, rank_tol = 1e-10)
    return get_homogeneous_matrix(HE)

def get_hyperellipsoid_from_homogeneous_matrix(Emat):
    An = (np.linalg.cholesky(Emat[:-1, :-1])).T
    center = np.linalg.solve(-Emat[:-1, :-1], Emat[-1, :-1])
    return Hyperellipsoid(An, center)

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