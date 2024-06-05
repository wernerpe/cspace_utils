from pydrake.all import (
                        MathematicalProgram,
                        CommonSolverOption,
                        GurobiSolver,
                        SolverOptions,
                        Solve,
                        )

import numpy as np

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


def compute_greedy_clique_partition(adj_mat, min_cliuqe_size, worklimit =100):
    cliques = []
    done = False
    adj_curr = adj_mat.copy()
    adj_curr = 1- adj_curr
    np.fill_diagonal(adj_curr, 0)
    ind_curr = np.arange(len(adj_curr))
    while not done:
        val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr, worklimit=worklimit) #solve_max_independet_set_KAMIS(adj_curr, maxtime = 5) #
        #non_max_ind_local = np.arange(len(adj_curr))
        #non_max_ind_local = np.delete(non_max_ind_local, ind_max_clique_local, None)
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        cliques.append(index_max_clique_global.reshape(-1))
        adj_curr = np.delete(adj_curr, ind_max_clique_local, 0)
        adj_curr = np.delete(adj_curr, ind_max_clique_local, 1)
        ind_curr = np.delete(ind_curr, ind_max_clique_local)
        if len(adj_curr) == 0 or len(cliques[-1])<min_cliuqe_size:
            done = True
    return cliques

