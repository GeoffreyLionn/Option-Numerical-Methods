import numpy as np
from scipy import stats

def black_scholes(S0, sigma, r, q, K, T, opt):
    
    '''
    Black Scholes Model to calculate the value of an option
    
    Args:
        S0 - intial price of underlying asset
        sigma - volatility
        r - risk-free rate
        q - dividend yield
        K - strike price
        T - time to maturity
        opt - 'call' or 'put'
        
    Returns the value of the option based on Black Scholes Model
    '''
    
    d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S0 * np.exp(-q * T) * stats.norm(0, 1).cdf(d1) - K * np.exp(-r * T) * stats.norm(0, 1).cdf(d2)
    elif opt == 'put':
        return K * np.exp(-r * T) * stats.norm(0, 1).cdf(-d2) - S0 * np.exp(-q * T) * stats.norm(0, 1).cdf(-d1)
    else:
        return 'Enter opt as call or put'

