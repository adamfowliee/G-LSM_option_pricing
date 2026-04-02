# used to find the cumulative distribution values
from scipy.stats import norm
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.stats import gmean
import pandas as pd

import glsm_geobasketcall
# calculating the price of the call and the put with put call parity

def geo_basket_bs_prices(Sigma_annual, S0_vec, K, r, T, dividend_array=None):
    """
    European call/put price for a geometric basket option under multivariate Black-Scholes.

    Sigma_annual: (d,d) covariance matrix of log-returns (annualized)
    S0_vec: (d,) spot prices
    K: strike (same units as S0)
    r: risk-free rate
    T: maturity in years
    dividend_array: (d,) dividend yields q_i (annualized), or None/0 for no dividends
    """
    S0_vec = np.asarray(S0_vec, dtype=float)
    d = S0_vec.size

    # geometric mean spot
    G0 = np.exp(np.mean(np.log(S0_vec)))

    # effective variance of log geometric mean
    ones = np.ones(d)
    sigmaG2 = (ones @ Sigma_annual @ ones) / (d**2)
    sigmaG = np.sqrt(sigmaG2)

    # average dividend yield
    if dividend_array is None:
        qbar = 0.0
    else:
        qbar = float(np.mean(np.asarray(dividend_array, dtype=float)))

    # average marginal variances
    sig_i2 = np.diag(Sigma_annual)
    mean_sig_i2 = float(np.mean(sig_i2))

    # effective dividend yield for the geometric basket process
    q_eff = qbar + 0.5 * mean_sig_i2 - 0.5 * sigmaG2

    # forward mean under Q
    F = G0 * np.exp((r - q_eff) * T)

    if sigmaG < 1e-14:
        # degenerate case: almost deterministic
        C = np.exp(-r*T) * max(F - K, 0.0)
        P = np.exp(-r*T) * max(K - F, 0.0)
        return C, P, q_eff, sigmaG

    d1 = (np.log(F / K) + 0.5 * sigmaG2 * T) / (sigmaG * np.sqrt(T))
    d2 = d1 - sigmaG * np.sqrt(T)

    C = np.exp(-r*T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    P = np.exp(-r*T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return C, P, q_eff, sigmaG




def bermudan_geo_basket_tree(p, Sigma_annual):
    """
    Bermudan option benchmark via 1D binomial tree on the geometric mean basket,
    with parameters matched to the multivariate BS geometric basket dynamics.
    """
    d = p['dim']
    S0_vec = np.asarray(p['S0'], dtype=float)
    G0 = np.exp(np.mean(np.log(S0_vec)))

    # effective variance of log geometric mean
    ones = np.ones(d)
    sigmaG2 = (ones @ Sigma_annual @ ones) / (d**2)
    sigmaG = np.sqrt(sigmaG2)

    # average dividend yield on components
    div_arr = p.get('dividend_array', None)
    if div_arr is None:
        qbar = 0.0
    else:
        qbar = float(np.mean(np.asarray(div_arr, dtype=float)))

    # average marginal variances
    mean_sig_i2 = float(np.mean(np.diag(Sigma_annual)))

    # effective dividend yield for geometric basket
    q_eff = qbar + 0.5 * mean_sig_i2 - 0.5 * sigmaG2

    # tree params
    N = int(p['numTimeStep'])
    T = float(p['expiration'])
    r = float(p['rate'])
    K = float(p['strike'])
    dt = T / N

    u = np.exp(sigmaG * np.sqrt(dt))
    d_down = 1.0 / u

    # risk-neutral up prob for GBM with carry (r - q_eff)
    p_up = (np.exp((r - q_eff) * dt) - d_down) / (u - d_down)
    df = np.exp(-r * dt)

    # terminal node prices
    S = G0 * (u ** np.arange(N, -N - 1, -2))

    if p['callput'] == 'call':
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # backward induction with Bermudan exercise at every step
    for i in range(N - 1, -1, -1):
        V = df * (p_up * V[:-1] + (1 - p_up) * V[1:])
        S = G0 * (u ** np.arange(i, -i - 1, -2))

        if p['callput'] == 'call':
            EV = np.maximum(S - K, 0.0)
        else:
            EV = np.maximum(K - S, 0.0)

        V = np.maximum(V, EV)

    return float(V[0])



etf_shrunk_cov_annual = np.array(pd.read_csv("data/etf_covariance_matrix_04032026.csv", index_col=0))
etf_prices = pd.read_csv("data/etf_prices_04032026.csv", index_col='Date')

# get current prices
S0_etf = etf_prices.iloc[-1, :]
S0_etf = np.array(S0_etf)

# get geometric average of the stocKprices
S0_etf_geo_mean = gmean(S0_etf)


r = 0.0375             # current BOE rate
div = 0                # no dividends on ETFs
T = 0.25               # exercise time
d = etf_shrunk_cov_annual.shape[0]       # number of assets

p_call = {}
p_call['rate'] = r     
p_call['dividend'] = div
p_call['dividend_array'] = None
p_call['expiration'] = T
p_call['dim'] = d
p_call['S0'] = S0_etf
p_call['volatility'] = None
p_call['correlation'] = None
p_call['covariance'] = etf_shrunk_cov_annual
p_call['numTimeStep'] = 50
p_call['callput'] = 'call'

p_put = p_call.copy()
p_put['callput'] = 'put'

# create df to store results
cols=["strike_mul", "strike", "BS_call", "BS_put", "glsm_call", "glsm_put", "bm_call", "bm_put"]
results_df = pd.DataFrame(columns=cols)

for m in np.linspace(0.9, 1.15, 8):
    print(m)
    # choose strike price
    K_etf = m * S0_etf_geo_mean

    p_call['strike'] = K_etf
    p_put['strike'] = K_etf

    C, P, q_eff, sigmaG = geo_basket_bs_prices(
        Sigma_annual=etf_shrunk_cov_annual,
        S0_vec=S0_etf,
        K=K_etf,
        r=r,
        T=T,
        dividend_array=None  # or p_call['dividend_array']
    )
    
    benchmark_call = bermudan_geo_basket_tree(p_call, etf_shrunk_cov_annual)
    benchmark_put  = bermudan_geo_basket_tree(p_put,  etf_shrunk_cov_annual)

    # Call the main function
    glsm_c, std_c = glsm_geobasketcall.main(p_call)
    glsm_p, std_p = glsm_geobasketcall.main(p_put)

    res = [m, K_etf, C, P, glsm_c, glsm_p, benchmark_call, benchmark_put]
    results_df.loc[len(results_df)] = res



results_df.to_csv("data/moneyness_test_2.csv")