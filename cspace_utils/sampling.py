import numpy as np

def point_in_regions(pt, regions):
    for r in regions:
        if r.PointInSet(pt.reshape(-1,1)):
            return True
    return False

def sample_in_union_of_polytopes(num_points, regions, aabb_limits, maxit = int(1e4), seed = 1234976512):
    #np.random.seed(seed)
    dim = regions[0].ambient_dimension()
    min = aabb_limits[0]
    max = aabb_limits[1]
    diff = max - min
    pts = np.zeros((num_points, dim))
    for i in range(num_points):
        for it in range(maxit):
            pt = min + np.random.rand(dim)*diff
            if point_in_regions(pt, regions):
                pts[i,:] = pt
                break
            if it == maxit-1:
                print("[sample_in_union_of_polytopes] NO POINT FOUND")
                return None   
    return pts