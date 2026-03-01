import numpy as np

def gen_paths_multi_bs(p, M):
    """
    Generate multiple asset paths using multivariate Black-Scholes model.
    
    Parameters:
    -----------
    p : dict
        Dictionary containing:
        - 'rate': interest rate(s)
        - 'dividend': dividend yield(s)
        - 'expiration': time to expiration
        - 'dim': number of assets/dimensions
        - 'S0': initial stock price(s)
        - 'volatility': volatility matrix
        - 'correlation': correlation matrix
        - 'numTimeStep': number of time steps
    M : int
        Number of simulation paths
    
    Returns:
    --------
    W : ndarray, shape (M, d, N)
        Brownian motion paths
    S : ndarray, shape (M, d, N)
        Stock price paths
    """
    r = p['rate']
    di = p['dividend']
    T = p['expiration']
    d = p['dim']
    S0 = p['S0']
    vol = p['volatility']
    P = p['correlation']
    N = p['numTimeStep']
    dt = T / N
    
    # Compute transformation
    cov = vol @ P @ vol.T
    eigenvalues, Q = np.linalg.eig(cov)
    
    # Sort eigenvalues in descending order
    ind = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[ind]
    Q = Q[:, ind]
    Lambda = np.diag(eigenvalues)
    

    ##### Maybe I should change this to make more efficient. Not sure yet
    # Compute drift parameter
    # mu = Q.T @ (r - di - 0.5 * np.square(vol) @ np.ones((d, 1)))
    # if vol is not diagonal then np.square(vol) doesn't work
    mu = Q.T @ (r - di - 0.5 * np.diag(cov).reshape(-1, 1))
    
    # Standard deviations (reshaped to row vector for broadcasting)
    SIG = np.sqrt(np.diag(Lambda)).reshape(1, -1)
    
    # Generate paths
    W = np.zeros((M, d, N))
    S = np.zeros((M, d, N))
    
    for k in range(N):
        if k == 0:
            W[:, :, k] = np.sqrt(dt) * np.random.randn(M, d)
        else:
            W[:, :, k] = W[:, :, k-1] + np.sqrt(dt) * np.random.randn(M, d)
        
        logprice = mu.T * (k + 1) * dt + SIG * W[:, :, k]
        S[:, :, k] = np.exp(logprice @ Q.T) * S0
    
    return W, S