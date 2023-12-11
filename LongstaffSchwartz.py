import numpy as np

def longstaff_schwartz(S0, sigma, r, q, K, T, opt, m, n, degree):
    
    
    '''
    Longstaff-Schwartz Least Squares Monte Carlo to calculate the value of an option
    
    Args:
        S0 - intial price of underlying asset
        sigma - volatility
        r - risk-free rate
        q - dividend yield
        K - strike price
        T - time to maturity
        opt - 'call' or 'put'
        m - number of time steps of each path
        n - number of simulations
        degree - highest order in polynomial fit
        
    Returns the value of the option based on Longstaff-Schwartz Least Squares Monte Carlo
    '''
    
    dt = T / m
    df = np.exp(-r * dt)

    # Stock Price Paths
    S = S0 * np.exp(np.cumsum((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * 
                              np.random.standard_normal((m+1, n)), axis=0))
    S[0] = S0

    # Initialization
    if opt == 'call':
        h = np.maximum(S - K, 0)
    elif opt == 'put':
        h = np.maximum(K - S, 0)
    else:
        return 'Enter opt as call or put'
    
    V = h[-1]
    
    # LSM Valuation
    for t in range(m - 1, 0, -1):
        
        # Only including in-the money for the regression
        stock = []
        value = []
        for i in range(len(V)):
            if V[i] > 0:
                stock.append(S[t, i])
                value.append(V[i] * df)
                
        # Regression fitting
        rg = np.polyfit(stock, value, degree)
        
        # Continuation values
        C  = np.polyval(rg, S[t])
        
        # Early exercise decision
        V  = np.where(h[t] > C, h[t], V * df)

    V0 = df * V
    return np.sum(V0) / n

