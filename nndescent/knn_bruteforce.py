import nndescent.distances as nnd_dist
import numpy as np
from numba import njit
from nndescent.utils import *

class KNN_Bruteforce:


    def __init__(self, data, metric="cosine", n_neighbors=30):
        self.data = data
        self.dist = nnd_dist.named_distances[metric]
        self.n_points = data.shape[0]


    def query(self, query_points, k=15):
        n_query_points = query_points.shape[0]
        neighbors, distances = init_heap(n_query_points, k)
        compute_knn(self.data, query_points, neighbors, distances, self.dist)
        return neighbors, distances



@njit
def init_heap(n_points, n_neighbors):
    neighbors = np.full((n_points, n_neighbors), -1, dtype=np.int32)
    distances = np.full((n_points, n_neighbors), np.infty, dtype=np.float32)
    return (neighbors, distances)


        
@njit
def compute_knn(data, query_points, neighbors, distances, dist):
    
    n_points = data.shape[0]
    n_query_points = query_points.shape[0]

    for u in range(n_query_points):
        for v in range(n_points):
            heap_push(distances[u], neighbors[u], dist(query_points[u], data[v]), v)
        
        
