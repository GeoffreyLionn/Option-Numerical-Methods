import numpy as np

def implicit_finite_diff(S0, sigma, r, q, K, T, opt, m, n):
    
    '''
    Implicit Finite Difference to calculate the value of an option
    
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
        
    Returns the value of the option based on implicit finite difference
    '''
    
    dt = T / m
    Smax = 2 * S0
    dS = Smax / n
    grid = np.zeros((m + 1, n + 1))
    S = np.arange(0, Smax + 0.01, dS)
    
    def a(j):
        return 0.5 * dt * ((r - q) * j - (sigma * j) ** 2)

    def b(j):
        return 1 + dt * ((sigma * j) ** 2 + r)

    def c(j):
        return -0.5 * dt * ((r - q) * j + (sigma * j) ** 2)
    
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
        
    # Matrix A
    A = np.zeros((n-1, n-1))
    for i in range(n-1):
        if i != 0:
            A[i, i-1] = a(i+1)
        if i != n-2:
            A[i, i+1] = c(i+1)
        A[i, i] = b(i+1)
    
    a0 = a(1)
    cn = c(n-1)
    
    # Iterating through the grid
    for i in range(m-1, -1, -1):
        f = grid[i+1, 1:-1]
        f[0] = f[0] - (a0 * grid[i, 0])
        f[-1] = f[-1] - (cn * grid[i, n])
        grid[i, 1:-1] = np.linalg.solve(A, f)
        
    if n % 2 == 0:
        return grid[0, n // 2]
    return grid[0, n // 2 - 1]

