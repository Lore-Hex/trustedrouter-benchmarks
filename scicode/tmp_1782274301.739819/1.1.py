import numpy as np



# Background: The Conjugate Gradient (CG) method is an iterative algorithm for solving 
# symmetric positive-definite linear systems Ax = b. It is based on minimizing the quadratic 
# form f(x) = (1/2)x^T A x - b^T x, which is minimized when Ax = b. The key insight is that 
# the sequence of search directions (conjugate directions) are orthogonal with respect to the 
# A-inner product, meaning p_i^T A p_j = 0 for i ≠ j. This ensures convergence in at most 
# M iterations for an M×M matrix (in exact arithmetic). The algorithm maintains:
# 1. Residual r = b - Ax (gradient of the quadratic form)
# 2. Search direction p (initialized to the negative gradient)
# 3. Step size α computed to minimize along the search direction
# 4. Update coefficient β computed using the Polak-Ribière-Polyak formula to maintain conjugacy
# The method terminates when ||r|| < tol, indicating convergence to the solution.

def cg(A, b, x, tol):
    '''Inputs:
    A : Matrix, 2d array size M * M
    b : Vector, 1d array size M
    x : Initial guess vector, 1d array size M
    tol : tolerance, float
    Outputs:
    x : solution vector, 1d array size M
    '''
    
    # Compute initial residual r = b - Ax
    r = b - np.dot(A, x)
    
    # Initialize search direction as negative gradient (residual direction)
    p = r.copy()
    
    # Compute initial residual norm squared
    rsold = np.dot(r, r)
    
    # Iterate until convergence
    while np.sqrt(rsold) > tol:
        # Compute step size α = r^T r / (p^T A p)
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        
        # Update solution: x = x + α * p
        x = x + alpha * p
        
        # Update residual: r = r - α * A * p
        r = r - alpha * Ap
        
        # Compute new residual norm squared
        rsnew = np.dot(r, r)
        
        # Compute Polak-Ribière-Polyak coefficient β = r_new^T r_new / r_old^T r_old
        beta = rsnew / rsold
        
        # Update search direction: p = r + β * p
        p = r + beta * p
        
        # Update residual norm for next iteration
        rsold = rsnew
    
    return x


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('1.1', 3)
target = targets[0]

n = 7
h = 1.0/n
diagonal = [2/h for i in range(n)]
diagonal_up = [-1/h for i in range(n-1)]
diagonal_down = [-1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
x0 = np.zeros(n)
tol = 10e-5
assert np.allclose(cg(A, b, x0,tol), target)
target = targets[1]

n = 7
h = 1.0/n
diagonal = [1/h for i in range(n)]
diagonal_up = [-9/h for i in range(n-1)]
diagonal_down = [-9/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
A[:, 0] = 0
A[0, :] = 0
A[0, 0] = 1/h
b = np.array([0.1,0.1,0.0,10,0.0,0.1,0.1])
x0 = np.zeros(n)
maxIter = 200
tol = 10e-7
assert np.allclose(cg(A, b, x0,tol), target)
target = targets[2]

n = 7
h = 1.0/n
diagonal = [1/h for i in range(n)]
diagonal_up = [-0.9/h for i in range(n-1)]
diagonal_down = [-0.9/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
b = np.array([0.1,10.1,0.0,0.5,0.2,0.3,0.5])
x0 = np.zeros(n)
maxIter = 500
tol = 10e-7
assert np.allclose(cg(A, b, x0,tol), target)
