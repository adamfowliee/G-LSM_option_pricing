import numpy as np
from .generate_poly_basis_1d import generate_poly_basis_1d



def generate_poly_hermite(type_str, I, grid, scale):
    """
    Generate polynomial basis using Hermite polynomials.
    
    Parameters:
    -----------
    type_str : str
        Type of polynomial basis
    I : ndarray, shape (N, d)
        Index matrix for polynomial degrees
    grid : ndarray, shape (M, d)
        Grid points (column vector per dimension)
            all points in brownian motion tensor for current timestep k
    scale : float
        Scaling parameter
            k * dt
    Returns:
    --------
    A : ndarray, shape (M, N)
        Polynomial basis matrix
    """
    N, d = I.shape  # N: number of basis functions, d: dimension
    M = grid.shape[0]  # M: number of grid points
    A = np.zeros((M, N))
    
    order = np.max(I)  # maximum polynomial degree
    P1 = [None] * d  # list to store 1D basis
    
    # Generate 1D polynomial basis for each dimension
    for j in range(d):
        yy = grid[:, j]
        P1[j] = generate_poly_basis_1d(type_str, order, yy, scale)  # M-by-(order+1) matrix
    
    # Assemble d-dimensional basis by tensor product
    for n in range(N):
        P_all = np.zeros((M, d))
        for j in range(d):
            P_all[:, j] = P1[j][:, I[n, j]]
        A[:, n] = np.prod(P_all, axis=1)
    
    return A