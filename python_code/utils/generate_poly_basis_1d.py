import numpy as np
import math

def generate_poly_basis_1d(type_str, order, grid, scale):
    """
    Generate 1D polynomial basis functions.
    
    Parameters:
    -----------
    type_str : str
        Type of polynomial basis ('chebyshev', 'hermite', 'norm_hermite')
    order : int
        Maximum polynomial order
    grid : ndarray
        Grid points
    scale : float or list
        Scaling parameter(s)
    
    Returns:
    --------
    A : ndarray, shape (len(grid), order+1)
        Polynomial basis matrix
    """
    grid = np.asarray(grid).flatten()
    A = np.zeros((len(grid), order + 1))
    
    if type_str == 'chebyshev':
        # Chebyshev polynomials of the first kind T_n(x)
        # T_n(x) = cos( n acos(x) )
        domain = scale
        xmin = domain[0]
        xmax = domain[1]
        
        # Shift grid to [-1, 1]
        grid = (grid - xmin) * 2 / (xmax - xmin) - 1
        
        A[:, 0] = 1
        for i in range(1, order + 1):
            A[:, i] = np.cos(i * np.arccos(grid))
        
        # Set out-of-domain values to 0
        idx = (grid < -1) | (grid > 1)
        A[idx, :] = 0
    
    elif type_str == 'hermite':
        # Generalized Hermite polynomials
        # H_0(x) = 1, H_1(x) = x
        # For n >= 2, H_n(x) = x H_{n-1}(x) - (n-1) H_{n-2}(x)
        # H_n^t(x) = t^(n/2) H_n(x/sqrt(t))
        t = scale
        yy = grid / np.sqrt(t)
        
        A[:, 0] = 1
        A[:, 1] = yy
        for i in range(2, order + 1):
            A[:, i] = yy * A[:, i - 1] - (i - 1) * A[:, i - 2]
        
        A = A * np.power(t, np.arange(order + 1) / 2)
    
    elif type_str == 'norm_hermite':
        # Normalized generalized Hermite polynomials
        # H_n^t(x) * t^(-n/2) * (n!)^(-0.5)
        t = scale
        yy = grid / np.sqrt(t)
        
        A[:, 0] = 1
        A[:, 1] = yy
        for i in range(2, order + 1):
            A[:, i] = yy * A[:, i - 1] - (i - 1) * A[:, i - 2]
        
        factorials = np.array([math.factorial(i) for i in range(order + 1)])
        A = A / np.sqrt(factorials)
    
    else:
        print('wrong type')
    
    return A