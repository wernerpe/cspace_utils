from pydrake.all import (
                        MathematicalProgram,
                        CommonSolverOption,
                        GurobiSolver,
                        SolverOptions,
                        Solve,
                        )

import numpy as np
from pydrake.all import MaxCliqueSolverViaGreedy
import networkx as nx

def solve_max_independent_set_integer(adj_mat, worklimit = 1000):
    n = adj_mat.shape[0]
    if n == 1:
        return 1, np.array([0])
    prog = MathematicalProgram()
    v = prog.NewBinaryVariables(n)
    prog.AddLinearCost(-np.sum(v))
    for i in range(0,n):
        for j in range(i,n):
            if adj_mat[i,j]:
                prog.AddLinearConstraint(v[i] + v[j] <= 1)

    solver_options = SolverOptions()
    #solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    solver_options.SetOption(GurobiSolver.id(), 'WorkLimit', worklimit)
    result = Solve(prog, solver_options=solver_options)
    return -result.get_optimal_cost(), np.nonzero(result.GetSolution(v))[0]


def compute_greedy_clique_partition(adj_mat, min_clique_size, worklimit =100):
    cliques = []
    done = False
    adj_curr = adj_mat.copy()
    adj_curr = 1- adj_curr
    np.fill_diagonal(adj_curr, 0)
    ind_curr = np.arange(len(adj_curr))
    while not done:
        print(f"[CCMIP] remaining nodes {adj_curr.shape[0]}")
        val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr, worklimit=worklimit) #solve_max_independet_set_KAMIS(adj_curr, maxtime = 5) #
        #non_max_ind_local = np.arange(len(adj_curr))
        #non_max_ind_local = np.delete(non_max_ind_local, ind_max_clique_local, None)
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        cliques.append(index_max_clique_global.reshape(-1))
        adj_curr = np.delete(adj_curr, ind_max_clique_local, 0)
        adj_curr = np.delete(adj_curr, ind_max_clique_local, 1)
        ind_curr = np.delete(ind_curr, ind_max_clique_local)
        if len(adj_curr) == 0 or len(cliques[-1])<min_clique_size:
            done = True
    return cliques

def double_greedy_clique_cover(adjacency_matrix, min_clique_size=8):
    cliques = []
    done = False
    adj_curr = adjacency_matrix.copy()
    ind_curr = np.arange(adj_curr.shape[0])
    while not done:
        print(f"[CCDGDRAKE] remaining nodes {adj_curr.shape[0]}")
        solver = MaxCliqueSolverViaGreedy()
        part_of_clique = solver.SolveMaxClique(adj_curr)
        ind_max_clique_local = np.where(part_of_clique)[0]
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        cliques.append(index_max_clique_global.reshape(-1))
        adj_curr = adj_curr[~part_of_clique][:, ~part_of_clique]
        ind_curr = ind_curr[~part_of_clique]
        if adj_curr.shape[0] == 0 or len(cliques[-1]) < min_clique_size:
            done = True
    return cliques

def adjacency_matrix_to_nxgraph(adj_matrix):
    G = nx.Graph()
    n = len(adj_matrix)
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    return G

def find_clique_cover_nx(G, min_clique_size = 8):
    clique_cover = []
    remaining_nodes = set(G.nodes())
    
    while remaining_nodes:
        print(f"[CCNXMAXCLIQUE] remaining nodes {len(remaining_nodes)}")
        # Find a maximal clique in the remaining graph
        clique = nx.approximation.max_clique(G.subgraph(remaining_nodes))
        clique_cover.append(list(clique))
        remaining_nodes -= set(clique)
        if len(clique_cover[-1])< min_clique_size:
            break
    return clique_cover

def double_greedy_clique_cover_nx(adjacency_matrix, min_clique_size = 8):
    nxg = adjacency_matrix_to_nxgraph(adjacency_matrix)
    return find_clique_cover_nx(nxg, min_clique_size)  

from typing import Union
from scipy.sparse import csc_matrix

def greedy_clique_cover(adjacency_matrix: Union[np.ndarray, csc_matrix], 
                        min_clique_size: int = 1, 
                        approach: str = 'dr', 
                        worklimit_mip: int = 100):
    assert approach in ['dr', 'nx', 'mip']
    '''
    Computes a trucnated clique cover (may not cover all nodes if min clique 
    size is larger than 1). 
    
    All clique covers are computed greedily using the iterative 
    clique removal strategy.

    dr: drake MaxCliqueSolverViaGreedy
    nx: networkx approximation max_clique
    mip: uses gurobi and drake to compute the maximum clique at every step
    '''
    
    if approach == 'dr':
        return double_greedy_clique_cover(adjacency_matrix, min_clique_size)
    if approach == 'nx':
        if isinstance(adjacency_matrix, csc_matrix):
            adjacency_matrix = adjacency_matrix.toarray()
        return double_greedy_clique_cover_nx(adjacency_matrix, min_clique_size)
    if approach == 'mip':
        if isinstance(adjacency_matrix, csc_matrix):
            adjacency_matrix = adjacency_matrix.toarray()
        return compute_greedy_clique_partition(adjacency_matrix, 
                                               min_clique_size,
                                               worklimit=worklimit_mip)
    raise ValueError("Invalid approach")