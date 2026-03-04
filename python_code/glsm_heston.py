import numpy as np
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent / 'utils'
sys.path.insert(0, str(utils_path))

from utils.gen_paths_heston_logscale import gen_paths_heston_logscale
from utils.gen_poly_basis_grad import gen_poly_basis_grad
from utils.hyperbolic_cross_indices import hyperbolic_cross_indices


# Price Bermudan put option under Heston model using G-LSM
# Reference:
# [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
#     expansions for pricing and hedging high-dimensional American options.

# Parameter ref: test No. 4 in the paper by Fang Fang and Cornelis W. Oosterlee,
#                "A Fourier-Based Valuation Method for Bermudan and Barrier
#                Options under Heston's Model"
# S0 = [8, 9, 10, 11, 12]
# V_ref_2011 = [2.000000, 1.107621, 0.520030, 0.213677, 0.082044]


class HestonParams:
    """Container for Heston model parameters."""
    def __init__(self):
        # self.S0 = 12
        # self.strike = 10
        # self.v0 = 0.0625
        # self.rate = 0.1
        # self.dividend = 0
        # self.rho = 0.1
        # self.kappa = 5
        # self.theta = 0.16
        # self.nu = 0.9
        # self.expiration = 0.25
        # self.numTimeStep = 50

        self.S0 = 6881.62       # Spot Price (March 2, 2026)
        self.strike = 6900      # Strike Price (ATM)
        self.v0 = 0.0225        # Initial Variance (VIX ~15%)
        self.rate = 0.036       # Risk-free Rate (3.6%)
        self.dividend = 0.012   # Dividend Yield (1.2%)
        self.rho = -0.7         # Correlation (Leverage effect)
        self.kappa = 3.0        # Mean Reversion
        self.theta = 0.04       # Long-run Variance
        self.nu = 0.4           # Vol of Vol
        self.expiration = 0.25  # Time to maturity (3 months)
        self.numTimeStep = 50   # Simulation steps


def main():
    """Main script for G-LSM Bermudan option pricing."""
    
    p = HestonParams()
    M = 10000
    order = 20
    I = hyperbolic_cross_indices(2, order)
    Nbasis = I.shape[0]
    
    # Running parameters
    num_trials = 10
    file_name = (f'heston_GLSM_So{int(p.S0)}_M{M}_Nb{Nbasis}_trials{num_trials}')
    V0_vals = np.zeros(num_trials)
    
    # Run and save
    for t in range(num_trials):
        V0_vals[t] = run_heston(p, M, I)
        print(f'run trial no.{t+1}, price = {V0_vals[t]:.4f}')
        print('---------------------------------------------')
    
    mean_price = np.mean(V0_vals)
    print(f'\nMean price: {mean_price:.6f}')
    print(f'Std dev: {np.std(V0_vals):.6f}')
    
    # Optionally save results
    # np.savez(f'data/{file_name}.npz', prices=V0_vals)


def run_heston(p, M, I):
    """
    Price a Bermudan put option under Heston model using G-LSM.
    
    Parameters:
    -----------
    p : HestonParams
        Heston model parameters
    M : int
        Number of Monte Carlo paths
    I : np.ndarray
        Hyperbolic cross indices for basis selection
        
    Returns:
    --------
    V0 : float
        Option price at t=0
    """
    
    S0 = p.S0
    K = p.strike
    r = p.rate
    rho = p.rho
    nu = p.nu
    T = p.expiration
    N = p.numTimeStep
    dt = T / N
    tau = (N - 1) * np.ones(M, dtype=int)  # Initialize to last valid index (0-based=int)
    
    # Generate paths
    Wpaths, Xpaths = gen_paths_heston_logscale(p, M)
    
    # Terminal payoff: exp(-r*T) * max(K - S0*exp(log_price), 0)
    payoff = np.exp(-r * T) * np.maximum(K - S0 * np.exp(Xpaths[:, 0, N-1]), 0)


    domain_logv = np.array([-7, np.log(0.8)])
    # domain_logv = np.array([Xpaths[:, 1, :].min() - 0.5, Xpaths[:, 1, :].max() + 0.5])
    poly_type = ['norm_hermite', 'chebyshev']
    
    # Backward induction through time steps
    for k in range(N-2, -1, -1):
        # Compute standard deviation of log-price at time step k
        xstd = np.std(Xpaths[:, 0, k])
        scale = [xstd, domain_logv]
        
        # Generate basis functions and gradients
        # Note: I.T transposes the index matrix to match expected dimensions
        A1, G = gen_poly_basis_grad(poly_type, I.T, Xpaths[:, :, k], scale)
        
        # Compute Brownian motion increments
        dW1 = Wpaths[:, 0, k+1] - Wpaths[:, 0, k]
        dW2 = Wpaths[:, 1, k+1] - Wpaths[:, 1, k]
        
        # Compute A2 term (gradient-enhanced basis)
        # Includes drift terms from gradient of price and log-variance
        sqrt_vol = np.exp(Xpaths[:, 1, k] / 2)
        inv_sqrt_vol = np.exp(-Xpaths[:, 1, k] / 2)
        
        combined_vol = (rho * dW1 + np.sqrt(1 - rho**2) * dW2) * sqrt_vol 
        A2 = combined_vol[:, np.newaxis] * G[0]
        A2 = A2 + ((nu * dW1 * inv_sqrt_vol)[:, np.newaxis] * G[1])
        
        # Solve least squares problem: (A1 + A2) * beta = payoff
        A_total = A1 + A2
        beta, _, _, _ = np.linalg.lstsq(A_total, payoff, rcond=None)
        
        # Compute continuation value
        CV = A1 @ beta
        
        # Compute exercise value at current time step
        EV = np.exp(-r * k * dt) * np.maximum(K - S0 * np.exp(Xpaths[:, 0, k]), 0)
        
        # Update stopping times and payoffs
        # Exercise if continuation value < exercise value and exercise value > 0
        idx = (CV < EV) & (EV > 0)
        tau[idx] = k
        payoff[idx] = EV[idx]
        payoff[~idx] = CV[~idx]
    
    # Compute price at t=0 using stopping times
    vv = np.zeros(M)
    for m in range(M):
        vv[m] = np.exp(-r * tau[m] * dt) * max(K - S0 * np.exp(Xpaths[m, 0, tau[m]]), 0)
    
    V0 = np.mean(vv)
    return V0


if __name__ == '__main__':
    main()
