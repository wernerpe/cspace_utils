import numpy as np
from pydrake.all import (MathematicalProgram, 
                         SolverOptions, 
                         CommonSolverOption, 
                         GurobiSolver,
                         Solve,
                         ge,
                         eq)
from typing import Union
from scipy.sparse import csc_matrix

def cutting_planes(clique, points, tol=1e-5):
    
    n, d = points.shape
    m = len(clique)
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(m)
    prog.AddLinearConstraint(ge(x, 0))
    prog.AddLinearConstraint(sum(x) == 1)
    constr = prog.AddLinearConstraint(eq(x @ points[clique], np.zeros(d)))
    cost = prog.AddLinearCost(sum(x))
    solver = GurobiSolver()
    
    # n x dim
    clique_points = points[clique]

    counterexamples = []
    for i in (j for j in range(n) if j not in clique):
        # next line is confusing but seems to do the right thing  
        # distances = np.linalg.norm(clique_points - points[i], axis = 1)
        # np.random.randn(m)
        constr.evaluator().UpdateUpperBound(points[i])
        # cost.evaluator().UpdateCoefficients(distances)
        cost.evaluator().UpdateCoefficients(np.random.randn(m))
        result = solver.Solve(prog)
        if result.is_success():
            x_opt = result.GetSolution(x)
            nonzeros = [clique[j] for j, xj in enumerate(x_opt) if xj > tol]
            counterexamples.append((i, nonzeros))   
    return counterexamples

def max_clique_iterative_cvx_h_constraint(adj_mat, graph_vertices, c = None):
    assert adj_mat.shape[0] == len(graph_vertices)
    n = adj_mat.shape[0]
    if c is None:
        c = np.ones((n,))
    prog = MathematicalProgram()
    v = prog.NewBinaryVariables(n)
    prog.AddLinearCost(-np.sum(c*v))
    for i in range(0,n):
        for j in range(i+1,n):
            if adj_mat[i,j] == 0:
                prog.AddLinearConstraint(v[i] + v[j] <= 1)
    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)
    #solver = GurobiSolver()
    num_cuts_iters = 0
    while True:
        # This should be a solver callback...
        result = Solve(prog, solver_options=solver_options)
        print(result.get_solver_details().optimizer_time)
        v_opt = result.GetSolution(v)
        clique = list(np.where(v_opt > .5)[0])
        print(f'######### current clique length {len(clique)}')

        cuts = cutting_planes(clique, graph_vertices)
        if len(cuts) == 0:
            print(f"######## Num cut iters {num_cuts_iters}")
            break
        print(f"num cuts {len(cuts)}")
        for i, nonzeros in cuts:
            prog.AddLinearConstraint(v[i] >= sum(v[nonzeros]) - len(nonzeros) + 1)
            
        num_cuts_iters +=1
        print(f"######## Num cut iters {num_cuts_iters}")
    return -result.get_optimal_cost(), np.array(clique) 
