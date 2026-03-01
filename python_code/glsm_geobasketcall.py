import numpy as np
from scipy.sparse.linalg import cgs

# Import the converted utility functions
from utils.gen_paths_multi_bs import gen_paths_multi_bs
from utils.payoff_geo import payoff_geo
from utils.generate_poly_hermite import generate_poly_hermite
from utils.hyperbolic_cross_indices import hyperbolic_cross_indices


def run_geobaskput(p, M, order):
    """
    Run the G-LSM algorithm for Bermudan geometric basket option pricing.
    """
    type_str = 'norm_hermite'
    K = p['strike']
    r = p['rate']
    T = p['expiration']
    d = p['dim']
    N = p['numTimeStep']
    dt = T / N
    tau = N * np.ones(M, dtype=int)
    
    I = hyperbolic_cross_indices(d, order)
    Nbasis = I.shape[0]
    
    # Generate paths
    Wpaths, Spaths = gen_paths_multi_bs(p, M)
    valueMatrix = payoff_geo(Spaths, K, r, dt, p['callput'])
    
    # Local gradient indices: for each basis n and dimension j,
    # find the index of the polynomial with degree decreased in dimension j
    loc_grad = np.zeros((Nbasis, d), dtype=int)
    for n in range(Nbasis):
        for j in range(d):
            target = I[n, :].copy()
            target[j] = max(0, target[j] - 1)
            # Find matching row in I
            for i in range(I.shape[0]):
                if np.array_equal(I[i, :], target):
                    loc_grad[n, j] = i
                    break
    
    # Dynamic programming
    payoff = valueMatrix[:, N - 1].copy()
    
    for k in range(N - 1, 0, -1):
        scale = k * dt
        
        # Generate basis and gradient-enhanced basis
        # This is algortihm 4.2
        A1 = generate_poly_hermite(type_str, I, Wpaths[:, :, k], scale)
        A = A1.copy()
        
        for j in range(d):
            dW = Wpaths[:, j, k] - Wpaths[:, j, k - 1]
            for n in range(Nbasis):
                if I[n, j] >= 1:
                    A[:, n] = A[:, n] + dW * A1[:, loc_grad[n, j]] * np.sqrt(I[n, j] / scale)
        
        # Solve least squares problem using conjugate gradient
        ATA = A.T @ A / M
        ATb = A.T @ payoff / M
        beta, info = cgs(ATA, ATb)
        
        if info != 0:
            print(f'CG solver did not converge at step {k}')
        
        # Compute continuation value
        CV = A1 @ beta
        EV = valueMatrix[:, k]
        
        # Exercise decision: exercise if immediate payoff > continuation value
        idx = (CV < EV) & (EV > 0)
        tau[idx] = k
        payoff[idx] = EV[idx]
        payoff[~idx] = CV[~idx]
    
    # Compute price at t=0
    V0 = np.mean(valueMatrix[np.arange(M), tau - 1])
    
    return V0

def main():
    """
    Main script: price Bermudan geometric basket put option using G-LSM.
    """
    # Set parameters
    p = {}
    p['strike'] = 100
    p['rate'] = 0.03
    p['dividend'] = 0
    p['expiration'] = 0.25
    p['dim'] = 2
    p['S0'] = 100 * np.ones(p['dim'])
    p['volatility'] = np.diag(np.ones(p['dim'])) * 0.2
    p['correlation'] = 0.5 * np.eye(p['dim']) + 0.5 * np.ones((p['dim'], p['dim']))
    p['numTimeStep'] = 100
    p['callput'] = 'put'
    
    M = 1000
    order = 10
    I = hyperbolic_cross_indices(p['dim'], order)
    Nbasis = I.shape[0]
    
    # Running parameters
    num_trials = 10
    file_name = f"geobaskput_GLSM_d{p['dim']}_M{M}_order{order}_Nb{Nbasis}_trials{num_trials}"
    V0_vals = np.zeros(num_trials)
    
    # Run trials
    print(f'Number of basis functions: {Nbasis}')
    print('---------------------------------------------')
    
    for t in range(num_trials):
        V0_vals[t] = run_geobaskput(p, M, order)
        print(f'run trial no.{t + 1}, price = {V0_vals[t]:.4f}')
        print('---------------------------------------------')
    
    print(f'\nMean price: {np.mean(V0_vals):.4f}')
    print(f'Std dev:    {np.std(V0_vals):.4f}')
    
    # Optionally save results
    # np.savez(f'data/{file_name}.npz', V0_vals=V0_vals)

if __name__ == '__main__':
    main()