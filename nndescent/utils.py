import numpy as np
from numba import njit, prange
import random

@njit(parallel=True)
def make_heap(n_points, n_neighbors):
    indices = np.full((n_points, n_neighbors), -1, dtype=np.int32)
    distances = np.full((n_points, n_neighbors), np.infty, dtype=np.float32)
    flags = np.zeros((n_points, n_neighbors), dtype=np.uint8)
    return (indices, distances, flags)

@njit
def flagged_heap_push(priorities, indices, flags, p, i, f):
    if priorities[0] <= p:
        return 0
    
    size = priorities.shape[0]
    
    idx = 0
    while True:
        left_idx = idx * 2 + 1
        right_idx = left_idx + 1
        
        if left_idx >= size:
            break
        elif right_idx >= size:
            if priorities[left_idx] > p:
                swap_idx = left_idx
            else: break
        elif priorities[left_idx] >= priorities[right_idx]:
            if priorities[left_idx] > p:
                swap_idx = left_idx
            else: break
        else:
            if priorities[right_idx] > p:
                swap_idx = right_idx
            else: break

        priorities[idx] = priorities[swap_idx]
        indices[idx] = indices[swap_idx]
        flags[idx] = flags[swap_idx]
        
        idx = swap_idx

    priorities[idx] = p
    indices[idx] = i
    flags[idx] = f

    return 1

@njit
def checked_flagged_heap_push(priorities, indices, flags, p, i, f):
    if priorities[0] <= p:
        return 0
    
    size = priorities.shape[0]

    for j in range(size):
        if indices[j] == i:
            return 0
    
    idx = 0
    while True:
        left_idx = idx * 2 + 1
        right_idx = left_idx + 1
        
        if left_idx >= size:
            break
        elif right_idx >= size:
            if priorities[left_idx] > p:
                swap_idx = left_idx
            else: break
        elif priorities[left_idx] >= priorities[right_idx]:
            if priorities[left_idx] > p:
                swap_idx = left_idx
            else: break
        else:
            if priorities[right_idx] > p:
                swap_idx = right_idx
            else: break

        priorities[idx] = priorities[swap_idx]
        indices[idx] = indices[swap_idx]
        flags[idx] = flags[swap_idx]
        
        idx = swap_idx

    priorities[idx] = p
    indices[idx] = i
    flags[idx] = f

    return 1

@njit
def heap_push(priorities, indices, p, i):
    if priorities[0] <= p:
        return 0
    
    size = priorities.shape[0]
    
    idx = 0
    while True:
        left_idx = idx * 2 + 1
        right_idx = left_idx + 1
        
        if left_idx >= size:
            break
        elif right_idx >= size:
            if priorities[left_idx] > p:
                swap_idx = left_idx
            else: break
        elif priorities[left_idx] >= priorities[right_idx]:
            if priorities[left_idx] > p:
                swap_idx = left_idx
            else: break
        else:
            if priorities[right_idx] > p:
                swap_idx = right_idx
            else: break

        priorities[idx] = priorities[swap_idx]
        indices[idx] = indices[swap_idx]
        
        idx = swap_idx

    priorities[idx] = p
    indices[idx] = i

    return 1

