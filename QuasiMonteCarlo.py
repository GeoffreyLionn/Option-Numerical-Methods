import numpy as np
import pandas as pd
from scipy import stats

def quasi_monte_carlo(S0, sigma, r, q, K, T, opt, n):
    
    '''
    Quasi Monte-Carlo Simulation to calculate the value of an option
    
    Args:
        S0 - intial price of underlying asset
        sigma - volatility
        r - risk-free rate
        q - dividend yield
        K - strike price
        T - time to maturity
        opt - 'call' or 'put'
        n - number of simulations
        
    Returns the option payoff simulations in Pandas Series
    '''
    
    dist = stats.qmc.MultivariateNormalQMC(mean = [0], cov = [[1]])
    sample = dist.random(n)
    output = []
    for i in sample:
        ST = S0 * np.exp((r - q - sigma ** 2 / 2) * T + sigma * i[0] * np.sqrt(T))
        if opt == 'call':
            payoff = max(K - ST, 0) * np.exp(-r * T)
        elif opt == 'put':
            payoff = max(K - ST, 0) * np.exp(-r * T)
        else:
            return 'Enter opt as call or put'
        output.append(payoff)
    return pd.Series(output)

