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
    cov = p['covariance']
    N = p['numTimeStep']
    dt = T / N
    
    if cov == None:
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
    
    # Generate N+1 steps (Index 0 is t=0, Index N is t=T)
    W = np.zeros((M, d, N)) 
    S = np.zeros((M, d, N))
    
    # Set initial stock prices at t=0
    S[:, :, 0] = S0
    
    for k in range(N):
        if k == 0:
            W[:, :, k] = np.sqrt(dt) * np.random.randn(M, d)
        else:
            W[:, :, k] = W[:, :, k-1] + np.sqrt(dt) * np.random.randn(M, d)
        
        logprice = mu.T * (k + 1) * dt + SIG * W[:, :, k]
        S[:, :, k] = np.exp(logprice @ Q.T) * S0
    
    return W, S