import numpy as np
from scipy.special import factorial
from .generate_poly_basis_1d import generate_poly_basis_1d

def gen_poly_basis_grad(type_list, I, grid, scale):
    """
    Generate polynomial basis functions and their gradients using tensor product.
    
    Parameters:
    -----------
    type_list : list of str
        Types of polynomials for each dimension
    I : np.ndarray
        Index matrix of shape (d, N), where d is dimension and N is number of basis
    grid : np.ndarray
        Grid points of shape (M, d), column vectors for each dimension
    scale : list
        Scaling parameters for each dimension
        
    Returns:
    --------
    A : np.ndarray
        Basis function matrix of shape (M, N)
    G : list of np.ndarray (optional)
        Gradients for each dimension, each of shape (M, N)
    """
    
    d, N = I.shape  # get N (number of matrix columns) and d (dimension)
    M = grid.shape[0]  # get M (number of matrix rows)
    A = np.zeros((M, N))  # initialize A
    order = np.max(I)  # find maximum polynomial degree
    P1 = [None] * d  # store 1d basis
    
    # store 1d derivative
    Pd1 = [None] * d
    
    # first dim use type_list[0]
    t = scale[0]
    yy = grid[:, 0]
    P1[0] = generate_poly_basis_1d(type_list[0], order, yy, t)  # M-by-(order+1) matrix
    Pd1[0] = generate_poly_grad_1d(type_list[0], order, yy, t)  # M-by-(order+1) matrix
    
    # second dim use type_list[1]
    domain = scale[1]
    yy = grid[:, 1]
    P1[1] = generate_poly_basis_1d(type_list[1], order, yy, domain)
    Pd1[1] = generate_poly_grad_1d(type_list[1], order, yy, domain)  # M-by-(order+1) matrix
    
    # assemble 2d basis by tensor product
    for n in range(N):
        P_all = np.zeros((M, d))
        for j in range(d):
            P_all[:, j] = P1[j][:, I[j, n]]
        A[:, n] = np.prod(P_all, axis=1)
    
    # Compute gradients if needed
    G = [None] * d
    for j in range(d):
        G[j] = np.zeros((M, N))
        for n in range(N):
            Q_all = np.zeros((M, d))
            for k in range(d):
                if k == j:
                    Q_all[:, k] = Pd1[k][:, I[k, n]]
                else:
                    Q_all[:, k] = P1[k][:, I[k, n]]
            G[j][:, n] = np.prod(Q_all, axis=1)
    
    return A, G


def generate_poly_grad_1d(poly_type, order, grid, scale):
    """
    Generate gradients of 1D polynomial basis functions.
    
    Parameters:
    -----------
    poly_type : str
        Type of polynomial ('chebyshev', 'hermite', 'norm_hermite')
    order : int
        Maximum polynomial degree
    grid : np.ndarray
        Grid points (1D array)
    scale : float or tuple
        Scaling parameter(s) for the polynomial
        
    Returns:
    --------
    G : np.ndarray
        Gradient matrix of shape (len(grid), order+1)
    """
    
    m = len(grid)
    G = np.zeros((m, order + 1))
    
    if poly_type == 'chebyshev':
        # Gradient of Chebyshev polynomials of the first kind T_n(x)
        # T_n(x) = cos( n acos(x) )
        # d T_n(x) / dx = n U_{n-1}(x), ==> G(:,n+1)
        # where U_{n-1}(x) = 2x U_{n-2}(x) - U_{n-3}(x)
        # or U_{n-1}(x) = sin( n * theta ) ./ sin(theta)
        # with theta = acos(x)
        
        domain = scale
        xmin, xmax = domain[0], domain[1]
        grid_clipped = np.clip(grid, xmin, xmax)
        grid_normalized = (grid_clipped - xmin) * 2 / (xmax - xmin) - 1  # shift to [-1,1]
        
        U = np.zeros((m, order))
        U[:, 0] = 1.0
        if order > 1:
            U[:, 1] = 2 * grid_normalized
        
        G[:, 1] = 1.0
        if order > 1:
            G[:, 2] = 4 * grid_normalized
        
        for i in range(2, order):
            U[:, i] = 2 * grid_normalized * U[:, i-1] - U[:, i-2]
            G[:, i+1] = (i + 1) * U[:, i]
        
        # Set gradient to zero outside domain
        idx = (grid_normalized < -1) | (grid_normalized > 1)
        G[idx, :] = 0
        
    elif poly_type == 'hermite':
        # Gradient of generalized Hermite polynomials
        # H_n'(x) = n H_{n-1}(x)
        
        t = scale
        yy = grid / np.sqrt(t)
        
        A = np.zeros((m, order))
        A[:, 0] = 1.0
        if order > 1:
            A[:, 1] = yy
        
        G[:, 1] = 1.0
        if order > 1:
            G[:, 2] = 2 * yy
        
        for i in range(2, order):
            A[:, i] = yy * A[:, i-1] - (i-1) * A[:, i-2]
            G[:, i+1] = (i + 1) * A[:, i]
        
        # Scale by t^(n/2)
        powers = np.arange(order + 1) / 2.0
        G = G * (t ** powers)
        
    elif poly_type == 'norm_hermite':
        # Gradient of normalized generalized Hermite polynomials
        # H_n^t(x) * t^(-n/2) * (n!)^(-0.5)
        
        t = scale
        yy = grid / np.sqrt(t)
        
        A = np.zeros((m, order))
        A[:, 0] = 1.0
        if order > 1:
            A[:, 1] = yy
        
        G[:, 1] = 1.0
        if order > 1:
            G[:, 2] = 2 * yy
        
        for i in range(2, order):
            A[:, i] = yy * A[:, i-1] - (i-1) * A[:, i-2]
            G[:, i+1] = (i + 1) * A[:, i]
        
        G = G / np.sqrt(t)
        normalizer = np.sqrt(factorial(np.arange(order + 1)))
        G = G / normalizer
        
    else:
        raise ValueError(f'Invalid polynomial type: {poly_type}')
    
    return G
