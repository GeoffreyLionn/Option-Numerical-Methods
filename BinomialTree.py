import numpy as np

def binomial_tree(S0, sigma, r, q, K, T, opt, n):
    
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
        n - number of steps
        
    Returns the value of the option based on n-step binomial tree
    '''
        
    # Calculating Constants
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma *  np.sqrt(dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Asset prices at maturity
    ST = S0 * d ** (np.arange(n, -1, -1)) * u ** (np.arange(0, n + 1, 1))

    # Option values at maturity
    if opt == 'call':
        ST = np.maximum(ST - K, np.zeros(n + 1))
    elif opt == 'put':
        ST = np.maximum(K - ST, np.zeros(n + 1))
    else:
        return 'Enter opt as call or put'

    # Backward Steps
    for i in np.arange(n, 0, -1):
        ST = disc * (p * ST[1:i+1] + (1 - p) * ST[0:i])
    return ST[0]

