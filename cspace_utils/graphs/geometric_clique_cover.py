import numpy as np
from typing import Union
from scipy.sparse import csc_matrix
from pydrake.all import (MathematicalProgram, 
                         SolverOptions, 
                         CommonSolverOption, 
                         GurobiSolver,
                         Solve,
                         ge,
                         eq)
from .geom_clique_cover_helpers import max_clique_iterative_cvx_h_constraint
from .greedy_max_geom_clique import greedy_max_geometric_clique
from .max_geom_clique_with_ellipsoidal_convex_hull_constraint import compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint
from .greedy_max_geom_clique2 import greedy_max_geometric_clique2
def iterative_greedy_max_geom_clique_cover(adjacency_matrix,
                                           vertex_positions,
                                           min_gain_per_clique,
                                           use_two = True
                                           ):
    assert adjacency_matrix.shape[0] == vertex_positions.shape[1]
    cliques = []
    done = False
    ind_curr = np.arange(len(adjacency_matrix))
    c = np.ones((adjacency_matrix.shape[0],))
    while not done:
        if use_two:
            val, ind_max_clique = greedy_max_geometric_clique2(adjacency_matrix, 
                                                          vertex_positions.T,
                                                          c)
        else:
            val, ind_max_clique = greedy_max_geometric_clique(adjacency_matrix, 
                                                            vertex_positions.T,
                                                            c)
        c[ind_max_clique] = 0
        cliques.append(np.array(ind_max_clique).reshape(-1))
        if val< min_gain_per_clique or np.sum(c) == 0:
            done = True
    return cliques

def cutting_planes_geometric_clique_cover(adjacency_matrix,
                                          vertex_positions,
                                          min_gain_per_clique,
                                          worklimit_mip = 100):
    
    assert adjacency_matrix.shape[0] == vertex_positions.shape[1]
    cliques = []
    done = False
    ind_curr = np.arange(len(adjacency_matrix))
    c = np.ones((adjacency_matrix.shape[0],))
    while not done:
        val, ind_max_clique_local = max_clique_iterative_cvx_h_constraint(adjacency_matrix, 
                                                                          vertex_positions.T,
                                                                          c)
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        c[ind_max_clique_local] = 0
        cliques.append(index_max_clique_global.reshape(-1))
        if val< min_gain_per_clique or np.sum(c) == 0:
            done = True
    return cliques

def ellipsoid_geometric_clique_cover(adjacency_matrix,
                                     vertex_positions,
                                     min_gain_per_clique,
                                     worklimit_mip = 100):
    cliques, elliposids = compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint(adjacency_matrix,
                                                                                           vertex_positions,
                                                                                           min_gain_per_clique,
                                                                                           50,
                                                                                           1.1)
    return cliques, elliposids

def greedy_geometric_clique_cover(adjacency_matrix: Union[np.ndarray, csc_matrix], 
                                  vertex_positions: np.ndarray,
                                  min_gain_per_clique: int = 1, 
                                  approach: str = 'cutting', 
                                  worklimit_mip: int = 100):
    assert approach in ['greedy','greedy2', 'cutting', 'ellipsoid']
    #this is a sanity check, the second dimension is the ambient dimension of the space
    if vertex_positions.shape[1] != adjacency_matrix.shape[1]:
        raise ValueError(f"""You likely forgot to transpose the vertex positions. 
                         They are interpreted as {vertex_positions.shape[1]} 
                         points in {vertex_positions.shape[0]} dimensions.""")
    '''
    Computes a trucnated clique cover (may not cover all nodes if min gain per clique 
    is larger than 1). 
    
    All clique covers are computed greedily using the iterative 
    clique removal strategy. For geometric cliques the values of covered 
    candidates is set to zero.

    greedy: uses greedy max geometric clique algorithm
    cutting: uses iterative cutting approach
    ellipsoidal: uses approximated ellipsoidal decision boundaries. 
    '''
    
    if approach == 'greedy':
        if isinstance(adjacency_matrix, csc_matrix):
            adjacency_matrix = adjacency_matrix.toarray()
        return iterative_greedy_max_geom_clique_cover(adjacency_matrix, 
                                                      vertex_positions, 
                                                      min_gain_per_clique,
                                                      use_two=False), None
    if approach == 'greedy2':
        if isinstance(adjacency_matrix, csc_matrix):
            adjacency_matrix = adjacency_matrix.toarray()
        return iterative_greedy_max_geom_clique_cover(adjacency_matrix, 
                                                      vertex_positions, 
                                                      min_gain_per_clique,
                                                      use_two=True), None
    if approach == 'cutting':
        if isinstance(adjacency_matrix, csc_matrix):
            adjacency_matrix = adjacency_matrix.toarray()
        return cutting_planes_geometric_clique_cover(adjacency_matrix, 
                                                      vertex_positions, 
                                                      min_gain_per_clique,
                                                      worklimit_mip), None
    if approach == 'ellipsoid':
        if isinstance(adjacency_matrix, csc_matrix):
            adjacency_matrix = adjacency_matrix.toarray()
        return ellipsoid_geometric_clique_cover(adjacency_matrix, 
                                                vertex_positions.T, 
                                                min_gain_per_clique,
                                                worklimit_mip)
    raise ValueError("Invalid approach")