"""
Hyperbolic Cross Indices Generation

Reference:
[1] "Computation of Induced Orthogonal Polynomial Distributions", Akil Narayan,...
    arXiv:1704.08465 [math], http://arxiv.org/abs/1704.08465
    Github repo: https://github.com/akilnarayan/induced-distributions/tree/master
"""

import numpy as np
from itertools import combinations


def hyperbolic_cross_indices(d, k):
    """
    Generate hyperbolic cross indices.
    
    Parameters:
    -----------
    d : int
        Dimension
    k : int
        Parameter k
        
    Returns:
    --------
    a : ndarray of shape (N, d)
        Hyperbolic cross indices, where N is the number of generated indices
    """
    if d == 1:
        a = np.arange(0, k + 1).reshape(-1, 1)
        return a
    
    a = np.zeros((1, d), dtype=int)
    
    for q in range(d):
        temp = np.zeros((k, d), dtype=int)
        temp[:, q] = np.arange(1, k + 1)
        a = np.vstack([a, temp])
    
    pmax = int(np.floor(np.log(k + 1) / np.log(2)))
    
    for p in range(2, pmax + 1):
        combs = np.array(list(combinations(range(d), p)))
        possible_indices = np.ones((1, p), dtype=int)
        ind = 0
        
        while ind < possible_indices.shape[0]:
            # Add any possibilities that are children of possible_indices(ind,:)
            alph = possible_indices[ind, :]
            for q in range(p):
                temp = alph.copy()
                temp[q] = temp[q] + 1
                if np.prod(temp + 1) <= k + 1:
                    possible_indices = np.vstack([possible_indices, temp])
            ind += 1
        
        possible_indices = np.unique(possible_indices, axis=0)
        arow = a.shape[0]
        a = np.vstack([a, np.zeros((combs.shape[0] * possible_indices.shape[0], d), dtype=int)])
        
        for c in range(combs.shape[0]):
            i1 = arow
            i2 = arow + possible_indices.shape[0]
            a[i1:i2, combs[c, :]] = possible_indices
            arow = i2
    
    return a

print(hyperbolic_cross_indices(3,5))