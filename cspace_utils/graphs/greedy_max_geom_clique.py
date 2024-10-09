import numpy as np
from pydrake.all import VPolytope 
from scipy.sparse import csc_matrix

def update_candidates(points_to_add, cands, adm):
    new_candidates = []
    for cand in cands:
        add = True
        for pt in points_to_add:
            if not adm[pt, cand]:
                add = False
        if add and not cand in points_to_add:
            new_candidates.append(cand)
    return new_candidates

def remove_disproven_candidates(w_ki, cands):
    new_cands = []
    num = 0
    for w,c in zip(w_ki, cands):
        if len(w)==0 or not w[0]==-1:
            new_cands.append(c)
            num+=1
    print(num)
    return new_cands

def check_if_clique(query, admat):
    if len(query)==1:
        return True
    adred = admat[query, :]
    adred = adred[:, query]
    #ignore diagonal entires 
    return np.sum(adred) >= ((adred.shape[0])**2 - (adred.shape[0]))

def compute_wki_and_values(clique, adj_mat, candidates_sorted, pts):
    #return values, sets wki, redirect to other solution
    assert len(clique)
    
    for c in candidates_sorted:
        assert check_if_clique(clique+[c], adj_mat)
    # assert check_if_clique(clique, adj_mat)
    clique_pts = pts[clique]
    non_clique_mems = []
    for i in range(len(pts)):
        if not i in clique:
            non_clique_mems.append(i)

    v_ki = np.zeros(len(candidates_sorted))
    w_ki_sorted = []
    #spanning_point = -np.ones(len(candidates))
    for current_candidate_idx, i in enumerate(candidates_sorted):
        #check if i is already contained in a W_ki
        already_contained_and_valid = False
        for prev_cand_idx, w in enumerate(w_ki_sorted):
            if i in w:
                already_contained_and_valid = True
                break
        
        if already_contained_and_valid:
            #if this point is part of the best expansion it will be added regardless of value, add dummy element
            w_ki_sorted.append([])
            #set the value 
            #v_ki[i] = v_ki[candidates[clique_spanning_point_candidate_idx]]
            #spanning_point[current_candidate_idx] = clique_spanning_point_candidate_idx
        else:
            #now need to compute V_ki,W_ki
            cand_pt = pts[i]
            cand_vpoly = VPolytope(np.concatenate((clique_pts, cand_pt.reshape(1,-1)),axis = 0).T)
            pts_in_cvxh = [i]
            for pt_idx in non_clique_mems:
                if cand_vpoly.PointInSet(pts[pt_idx]) and not pt_idx==i:
                    pts_in_cvxh.append(pt_idx)
            
            is_valid_spanning_point = check_if_clique(clique+pts_in_cvxh, adj_mat)
            if is_valid_spanning_point:
                v_ki[current_candidate_idx] = len(pts_in_cvxh)
                w_ki_sorted.append(pts_in_cvxh)
            else:
                print('invalid_spanning_point')
                w_ki_sorted.append([-1])
    return w_ki_sorted, v_ki

import time 

def greedy_max_geometric_clique(adj_mat,
                                pts):
    
    if isinstance(adj_mat, csc_matrix):
        adj_mat = adj_mat.toarray()
    degrees = adj_mat.sum(axis=1)
    candidates = np.argsort(degrees)[::-1]
    current_clique = []
    #do N steps greedy
    for _ in range(pts.shape[1]):
        current_clique.append(candidates[0])
        candidates = update_candidates([candidates[0]], candidates, adj_mat)
    iter = 0
    
    while len(candidates):
        print(f"candidates {len(candidates)}")
        t1 = time.time()
        w_ki, v_ki = compute_wki_and_values(current_clique, adj_mat, candidates, pts)
        t2 = time.time()
        if np.all(v_ki ==0):
            break
        # print(f"wki {w_ki}")
        # print(f"vki {v_ki}")
        
        max_vki = np.max(v_ki)
        best_pts = np.where(v_ki==max_vki)[0]
        if len(best_pts>1):
            degrees_of_best_pts = [degrees[candidates[pt]] for pt in best_pts]
            best = np.argmax(degrees_of_best_pts)
            best_cand = best_pts[best]
        else:
            best_cand = best_pts[0]
        # point_to_add = candidates[best_cand]
        current_clique = current_clique + w_ki[best_cand]
        candidates = remove_disproven_candidates(w_ki, candidates)
        candidates = update_candidates(w_ki[best_cand], candidates, adj_mat)
        t3 = time.time()
        iter+=1
        break
    print(f"t12 {t2-t1}")
    print(f"t23 {t3-t2}")
    return current_clique