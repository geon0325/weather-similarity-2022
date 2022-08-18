import numpy as np
import nndescent.distances as nnd_dist
import random
from tqdm import tqdm
from numba import njit
from numba.experimental import jitclass
import numba.types as nt
from nndescent.utils import *
from timeit import default_timer as timer
import heapq

class NNDescent:
    """
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
    metric: string (optional, default="cosine")
    n_neighbors: int (optional, default=30)
    max_candidates: int (optional, default=min(60, n_neighbors))
    n_iters: int (optional, default=max(5, int(round(np.log2(data.shape[0])))))
    init_method: string (optinal, default="random")
    """

    def __init__(self, data, metric="cosine", n_neighbors=30, max_candidates=None, max_iters=None, init_method="random"):
        
        self.data = data
        
        self.dist = nnd_dist.named_distances[metric]
        dist = self.dist

        if max_candidates == None:
            max_candidates = min(60, n_neighbors)

        if max_iters == None:
            max_iters = max(5, int(round(np.log2(data.shape[0]))))

        n_points = data.shape[0]
        
        init_graph = make_heap(n_points, n_neighbors)
        
        if init_method == "random":
            init_random(init_graph, data, dist)
        else:
            raise
        
        self.graph = init_graph
        neighbors, distances, flags = self.graph

        old_candidates = np.full((n_points, max_candidates), -1, dtype=np.int32)
        new_candidates = np.full((n_points, max_candidates), -1, dtype=np.int32)
        old_priorities = np.full((n_points, max_candidates), np.infty, dtype=np.float32)
        new_priorities = np.full((n_points, max_candidates), np.infty, dtype=np.float32)
        

        for iter in range(max_iters):
            t1 = timer()
            compute_candidates(neighbors, flags,
                           old_candidates, new_candidates,
                           old_priorities, new_priorities)

            t2 = timer()

            num_pushes = update_from_candidates(data, neighbors, distances, flags,
                           old_candidates, new_candidates, dist)

            t3 = timer()
            print(f"iter: {iter}, num_pushes: {num_pushes}, step1: {t2-t1:.4f}, step2: {t3-t2}, total: {t3-t1}")
            if num_pushes == 0:
                break

    def query(self, query_points, k=15, epsilon=0.1):
        try:
            visited = self.visited
        except AttributeError:
            self.visited = init_visited(self.graph[0].shape[0])
            visited = self.visited
        
        return query(self.data, self.graph, query_points, k, visited, epsilon, self.dist)


@njit(
    locals={
        "d": nt.float32,
        "v": nt.int32
    }
)
def query(data, graph, query_points, k, visited, epsilon, dist):
    n_points , size = data.shape
    n_queries = query_points.shape[0]

    graph_indices = graph[0]
    n_neighbors = graph_indices.shape[1]

    distance_scale = 1.0 + epsilon

    res_indices = np.full((int(n_queries), int(k)), -1, dtype=np.int32)
    res_distances = np.full((int(n_queries), int(k)), np.infty, dtype=np.float32)

    candidates = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]

    for i in range(n_queries):

        cur_res_indices = res_indices[i]
        cur_res_distances = res_distances[i]

        query_point = query_points[i]
        visited[:] = 0

        candidates.clear()
        initial_points = set()
        while len(initial_points) < k:
            initial_points.add(random.randrange(n_points))

        for v in initial_points:
            d = dist(data[v], query_point)
            heapq.heappush(candidates, (d, v))
            heap_push(cur_res_distances, cur_res_indices, d, v)
            mark_visited(visited, v)
        
        distance_bound = cur_res_distances[0] * distance_scale

        while len(candidates) > 0:
            u_dist, u = heapq.heappop(candidates)
            if u_dist > distance_bound: break
            
            for j in range(n_neighbors):
                v = graph_indices[u, j]

                if is_visited(visited, v) == 0:
                    mark_visited(visited, v)
                    d = dist(data[v], query_point)
                    
                    if d < distance_bound:
                        heapq.heappush(candidates, (d, v))
                        heap_push(cur_res_distances, cur_res_indices, d, v)
                        distance_bound = cur_res_distances[0] * distance_scale
                        
        
    return (res_indices, res_distances)


@njit
def init_visited(n_points):
    visited_size = (n_points >> 3) + (1 if n_points & 0xff else 0)
    return np.zeros(visited_size, dtype=np.int8)

@njit
def is_visited(visited, v):
    return visited[v >> 3] & (1 << (v & 7))

@njit
def mark_visited(visited, v):
    visited[v >> 3] |= (1 << (v & 7))

@njit
def init_random(graph, data, dist):
    indices, distances, flags = graph
    n_points, n_neighbors = indices.shape
    
    random_neighbors = set()
    
    for u in range(n_points):
        random_neighbors.clear()
        while len(random_neighbors) < n_neighbors:
            v = random.randrange(n_points)
            if v != u:
                random_neighbors.add(v)
                flagged_heap_push(distances[u], indices[u], flags[u], dist(data[u], data[v]), v, np.uint8(1))

@njit(parallel=True)
def compute_candidates(neighbors, flags, old_candidates, new_candidates, old_priorities, new_priorities):
        
    n_points, n_neighbors = neighbors.shape
    max_candidates = old_candidates.shape[1]

    old_candidates.fill(-1)
    new_candidates.fill(-1)
    old_priorities.fill(np.infty)
    new_priorities.fill(np.infty)

    for u in range(n_points):

        for i in range(n_neighbors):
            v = neighbors[u, i]
            is_new = flags[u, i]
            p = random.random()

            if is_new:
                heap_push(new_priorities[u], new_candidates[u], p, v)
                heap_push(new_priorities[v], new_candidates[v], p, u)
            else:
                heap_push(old_priorities[u], old_candidates[u], p, v)
                heap_push(old_priorities[v], old_candidates[v], p, u)


    for u in range(n_points):
        for i in range(n_neighbors):
            v = neighbors[u, i]
            for j in range(max_candidates):
                if v == new_candidates[u, j]:
                    flags[u, i] = 0
                    break

@njit
def update_from_candidates(data, neighbors, distances, flags, old_candidates, new_candidates, dist):

    n_points, n_neighbors = neighbors.shape
    max_candidates = old_candidates.shape[1]

    num_push = 0
    
    for i in range(n_points):
        for j in range(max_candidates):
            u = new_candidates[i, j]
            if u != -1:
                for k in range(max_candidates):
                    v = new_candidates[i, k]
                    if v != -1 and u != v:
                        d = dist(data[u], data[v])
                        num_push += checked_flagged_heap_push(distances[u], neighbors[u], flags[u], d, v, np.int8(1))
                        num_push += checked_flagged_heap_push(distances[v], neighbors[v], flags[v], d, u, np.int8(1))
                    
                    v = old_candidates[i, k]
                    if v != -1 and u != v:
                        d = dist(data[u], data[v])
                        num_push += checked_flagged_heap_push(distances[u], neighbors[u], flags[u], d, v, np.int8(1))
                        num_push += checked_flagged_heap_push(distances[v], neighbors[v], flags[v], d, u, np.int8(1))

    return num_push
