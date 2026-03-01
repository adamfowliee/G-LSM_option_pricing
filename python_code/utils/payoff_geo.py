import numpy as np
from scipy.stats import gmean

def payoff_geo(Spaths, K, r, dt, callput):
    """
    Compute the payoff matrix at each timestep for all samples.
    This is for the geometric basket. Create a function like this 
    to make other exotic options if you want.
    Parameters:
    -----------
    Spaths : ndarray, shape (M, d, N)
        Stock price paths
    K : float
        Strike price
    r : float
        Interest rate
    dt : float
        Time step
    callput : str
        'put' or 'call'
    
    Returns:
    --------
    valueMatrix : ndarray, shape (M, N)
        Payoff matrix for all samples at each timestep
    """
    M, _, N = Spaths.shape
    valueMatrix = np.zeros((M, N))
    
    if callput == 'put':
        for k in range(N):
            geo_means = gmean(Spaths[:, :, k], axis=1)
            valueMatrix[:, k] = np.exp(-r * (k + 1) * dt) * np.maximum(K - geo_means, 0)
    elif callput == 'call':
        for k in range(N):
            geo_means = gmean(Spaths[:, :, k], axis=1)
            valueMatrix[:, k] = np.exp(-r * (k + 1) * dt) * np.maximum(geo_means - K, 0)
    
    return valueMatrix