import numpy as np
from pydrake.all import VPolytope
from typing import Union, List, Dict
from scipy.sparse import csc_matrix

def update_candidates(points_added, current_candidates, adj, vk=None):
    cands = []
    for c in current_candidates:
        if c in points_added:
            continue
        if vk is not None:
            if vk[c] ==-1:
                continue
        do_add = True
        for p in points_added:
            if not adj[p,c]:
                do_add = False
                break
        if do_add:
            cands.append(c)
    return cands

def pick_best_verts_to_add(vk):
    vals = [v['value'] for v in vk]
    best_spanning_point = np.argmax(vals)
    wki = vk[best_spanning_point]['wki']
    return wki

def check_if_clique(query, admat):
    if len(query)==1:
        return True
    adred = admat[query, :]
    adred = adred[:, query]
    #ignore diagonal entires 
    return np.sum(adred) >= ((adred.shape[0])**2 - (adred.shape[0]))

def update_vk(vk: List[Dict[str, Union[float, List[int]]]], 
              current_clique: List[int], 
              candidates: List[int],
              adjacency_matrix: np.ndarray,
              degrees: List[int], 
              pts : np.ndarray):
    
    current_best_value = 0
    
    #reset associated points
    for v in vk:
        v['wk'] = []
    
    non_clique_mems = [i for i in range(len(pts)) if i not in current_clique]

    clique_pts = (pts[current_clique]).T
    for c in candidates:
        #TODO this may be redundant?
        if not check_if_clique(current_clique + [c], adjacency_matrix):
            vk[c]['value'] = -1
            continue
        #if c was an invalid spanning point in a previous iteration it is still invalid
        if vk[c]['value'] == -1:
            continue
        #check if the point cannot add value
        if degrees[c] - len(current_clique) < current_best_value:
            vk[c]['value'] = 0
            continue
        # Check if c is already contained in one of the wki 
        # (and therefore automatically dominated)
        for v in vk:
            if c in v['wk']:
               vk[c]['value'] = 0
               continue

        #build wk_c for candidate c
        spanning_pt = pts[c].T
        spanning_vpoly = VPolytope(np.concatenate((clique_pts, spanning_pt), axis = 1))
        wk_c = [c]
        for i in non_clique_mems:
            if spanning_vpoly.PointInSet(pts[i]) and not i==c:
                wk_c.append(i)
        is_valid_spanning_point = check_if_clique(current_clique + wk_c)
        if is_valid_spanning_point:
            vk[c]['value'] = len(wk_c) - len(current_clique)
            if current_best_value <vk[c]['value']:
                current_best_value = vk[c]['value']
            vk[c]['wki'] = wk_c
        else:
            vk[c]['value'] = -1
# open ideas
# reevaluate traversal order at every iteration (distance centered at 
# current clique instead of mean of all points)
# make check if clique more efficient by using knowledge of the current clique (not likely to have a big impact.) 

def greedy_max_geometric_clique2(adj_mat,
                                pts,
                                c = None,
                                do_n_steps_greedy = False):
    assert adj_mat.shape[0] == pts.shape[0]
    #this only supports c = {0,1}^N
    if isinstance(adj_mat, csc_matrix):
        adj_mat = adj_mat.toarray()

    degrees = adj_mat.sum(axis=1)
    degrees = degrees / (np.max(degrees)+1)
    distance_to_mean = np.linalg.norm(pts - np.mean(pts, axis=0).reshape(1,-1), axis =1)
    distance_to_mean = distance_to_mean / (np.max(distance_to_mean) + 1e-6)
    #TODO reevaluete traversal order in every iteration
    cand_traversal_values = degrees + distance_to_mean
    candidates = np.argsort(cand_traversal_values)[::-1]
    degree_candidates = np.argsort(degrees)[::-1]
    if c is not None:
        assert len(c) == len(pts)
        #extract non zero candidates
        csorted = c[candidates]
        candidates = np.delete(candidates, np.where(csorted == 0)[0])
        degree_candidates = np.delete(degree_candidates, np.where(csorted == 0)[0])
    original_candidates = candidates.copy()
    current_clique = []
    
    #Do up to N steps greedy, probability of clique being non_geometric is zero if smapled randomly,
    # otherwise only add the candidate with the highest degree
    for _ in range(pts.shape[1]):
        current_clique.append(degree_candidates[0])
        candidates = update_candidates([current_clique[-1]], candidates, adj_mat)
        if len(candidates) ==0 or not do_n_steps_greedy:
            break
        # Only worry about updating candidates sorted by degrees if we do all n steps of greedy clique building
        degree_candidates = update_candidates([current_clique[-1]], degree_candidates, adj_mat)

    vk = [{'value':0, 'wki': []}]*len(pts)
    # reuse more work when computing the vk_i make vk_i dict {int, list[int]} set already covered vk[i] = 0
    while len(candidates):
        # update values of remaining candidates
        vk = update_vk(vk, 
                       current_clique, 
                       candidates,
                       adj_mat, 
                       degrees, 
                       pts)

        wk = pick_best_verts_to_add(vk)
        if not len(wk):
            break

        current_clique += wk

        # Update candidates, remove points that are disconnected from clique, 
        # also remove invalid spanning points. An invalid spanning point itself 
        # may be connected to the current clique, but its associated set of 
        # points may not form a clique with the current clique. I.e. there 
        # are disconnected points in the convex hull of the clique and the 
        # spanning point. 
        candidates = update_candidates(wk, candidates, adj_mat, vk)
        
    num_cands_added = 0
    for c in current_clique:
        if c in original_candidates:
            num_cands_added+=1
    return num_cands_added, current_clique 