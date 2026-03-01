import numpy as np


def gen_paths_heston_logscale(p, M):
    """
    Generate paths of log-price and log-variance using Heston model.
    
    Parameters:
    -----------
    p : object
        Parameter object with attributes:
        - v0: initial variance
        - rate: interest rate
        - dividend: dividend yield
        - rho: correlation between price and variance
        - kappa: mean reversion speed
        - theta: long-term variance
        - nu: volatility of variance
        - expiration: time to expiration
        - numTimeStep: number of time steps
    M : int
        Number of paths
        
    Returns:
    --------
    W : np.ndarray
        Brownian motion increments of shape (M, 2, N)
    X : np.ndarray
        Log-price and log-variance paths of shape (M, 2, N)
    """
    
    d = 2
    v0 = p.v0
    r = p.rate
    di = p.dividend
    rho = p.rho
    kappa = p.kappa
    theta = p.theta
    nu = p.nu
    T = p.expiration
    N = p.numTimeStep
    dt = T / N
    
    W = np.zeros((M, d, N))
    X = np.zeros((M, d, N))
    
    for k in range(N):
        if k == 0:
            W[:, :, 0] = np.sqrt(dt) * np.random.randn(M, d)
            X[:, 0, 0] = (r - di - 0.5 * v0) * dt + np.sqrt(v0) * (
                rho * W[:, 0, 0] + np.sqrt(1 - rho**2) * W[:, 1, 0]
            )
            vol = np.abs(
                v0 + kappa * (theta - v0) * dt + nu * np.sqrt(v0) * W[:, 0, 0]
            )
            X[:, 1, 0] = np.log(vol)
        else:
            dW = np.sqrt(dt) * np.random.randn(M, d)
            W[:, :, k] = W[:, :, k-1] + dW
            sqrtvol = np.sqrt(vol)
            X[:, 0, k] = X[:, 0, k-1] + (r - di - 0.5 * vol) * dt + sqrtvol * (
                rho * dW[:, 0] + np.sqrt(1 - rho**2) * dW[:, 1]
            )
            vol = np.abs(
                vol + kappa * (theta - vol) * dt + nu * sqrtvol * dW[:, 0]
            )
            X[:, 1, k] = np.log(vol)
    
    return W, X


# def gen_paths_heston_logscale_alternative(p, M):
#     """
#     Generate paths using direct Euler discretization of log-variance.
#     
#     This approach applies Itô's lemma to discretize log-variance directly,
#     which includes a drift correction term and guarantees positivity.
#     
#     Parameters:
#     -----------
#     p : object
#         Parameter object with attributes (same as gen_paths_heston_logscale)
#     M : int
#         Number of paths
#         
#     Returns:
#     --------
#     W : np.ndarray
#         Brownian motion increments of shape (M, 2, N)
#     X : np.ndarray
#         Log-price and log-variance paths of shape (M, 2, N)
#     """
#     
#     d = 2
#     v0 = p.v0
#     r = p.rate
#     di = p.dividend
#     rho = p.rho
#     kappa = p.kappa
#     theta = p.theta
#     nu = p.nu
#     T = p.expiration
#     N = p.numTimeStep
#     dt = T / N
#     
#     W = np.zeros((M, d, N))
#     X = np.zeros((M, d, N))
#     
#     for k in range(N):
#         if k == 0:
#             W[:, :, 0] = np.sqrt(dt) * np.random.randn(M, d)
#             X[:, 0, 0] = (r - di - 0.5 * v0) * dt + np.sqrt(v0) * (
#                 rho * W[:, 0, 0] + np.sqrt(1 - rho**2) * W[:, 1, 0]
#             )
#             # Itô's lemma correction: (kappa*theta - ν²/2)/v0
#             X[:, 1, 0] = np.log(v0) + (
#                 (kappa * theta - 0.5 * nu**2) / v0 - kappa
#             ) * dt + (nu / np.sqrt(v0)) * W[:, 0, 0]
#         else:
#             dW = np.sqrt(dt) * np.random.randn(M, d)
#             W[:, :, k] = W[:, :, k-1] + dW
#             vol = np.exp(X[:, 1, k-1])
#             X[:, 0, k] = X[:, 0, k-1] + (r - di - 0.5 * vol) * dt + np.exp(
#                 X[:, 1, k-1] / 2
#             ) * (rho * dW[:, 0] + np.sqrt(1 - rho**2) * dW[:, 1])
#             # Discretize log-variance directly with Itô correction
#             X[:, 1, k] = X[:, 1, k-1] + (
#                 (kappa * theta - 0.5 * nu**2) * np.exp(-X[:, 1, k-1]) - kappa
#             ) * dt + nu * np.exp(-X[:, 1, k-1] / 2) * dW[:, 0]
#     
#     return W, X
