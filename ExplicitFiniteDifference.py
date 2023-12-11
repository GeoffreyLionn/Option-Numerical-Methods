import numpy as np

def explicit_finite_diff(S0, sigma, r, q, K, T, opt, m, n):
    
    '''
    Explicit Finite Difference to calculate the value of an option
    
    Args:
        S0 - intial price of underlying asset
        sigma - volatility
        r - risk-free rate
        q - dividend yield
        K - strike price
        T - time to maturity
        opt - 'call' or 'put'
        m - number of partition points (time periods)
        n - number of partition points (stock price)
        
    Returns the value of the option based on explicit finite difference
    '''
    
    dt = T / m
    Smax = 2 * S0
    dS = Smax / n
    grid = np.zeros((m + 1, n + 1))
    S = np.arange(0, Smax + 0.01, dS)
    
    def a(j):
        return (-0.5 * (r - q) * j * dt + 0.5 * (sigma * j) ** 2 * dt) / (1 + r * dt)
    
    def b(j):
        return (1 - (sigma * j) ** 2 * dt) / (1 + r * dt)
    
    def c(j):
        return (0.5 * (r - q) * j * dt + 0.5 * (sigma * j) ** 2 * dt) / (1 + r * dt)
    
    
    # Setting boundaries
    for i in range(m + 1):
        if opt == 'call':
            grid[i, 0] = 0
            grid[i, n] = K
        elif opt == 'put':
            grid[i, 0] = K
            grid[i, n] = 0
        else:
            return 'Enter opt as call or put'
    
    for j in range(n + 1):
        if opt == 'call':
            grid[m, j] = max(S[j] - K, 0)
        elif opt == 'put':
            grid[m, j] = max(K - S[j], 0)
        
    # Iterating through the grid
    for i in range(m-1, -1, -1):
        for j in range(1, n):
            grid[i, j] = a(j) * grid[i+1, j-1] + b(j) * grid[i+1, j] + c(j) * grid[i+1, j+1]
            
    if n % 2 == 0:
        return grid[0, n // 2]
    return grid[0, n // 2 - 1]

