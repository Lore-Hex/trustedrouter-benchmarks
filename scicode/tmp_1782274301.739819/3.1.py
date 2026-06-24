import numpy as np



# Background: The Gauss-Seidel method is an iterative technique for solving linear systems Ax=b.
# It decomposes matrix A into its diagonal (D), lower triangular (L), and upper triangular (U) components,
# where A = D - L - U. For Gauss-Seidel, M = D - L and N = U, leading to the iteration:
# (D - L)x_k = Ux_{k-1} + b, or equivalently: x_k = (D - L)^{-1}(Ux_{k-1} + b)
# 
# The method updates each component using newly computed values (forward substitution on the lower triangle),
# which typically provides faster convergence than Jacobi. The iteration continues until the L2 norm of the
# difference between consecutive iterates ||x_k - x_{k-1}||_2 < epsilon.
# 
# The algorithm:
# 1. Initialize x = x0
# 2. Repeat until convergence:
#    - For each row i: x_i = (b_i - sum(A_ij*x_j for j<i) - sum(A_ij*x_j for j>i)) / A_ii
#    - Check if ||x_new - x_old||_2 < eps
# 3. Compute residual ||Ax - b||_2 and error ||x - x_true||_2

def GS(A, b, eps, x_true, x0):
    '''Solve a given linear system Ax=b Gauss-Seidel iteration
    Input
    A:      N by N matrix, 2D array
    b:      N by 1 right hand side vector, 1D array
    eps:    Float number indicating error tolerance
    x_true: N by 1 true solution vector, 1D array
    x0:     N by 1 zero vector, 1D array
    Output
    residual: Float number shows L2 norm of residual (||Ax - b||_2)
    errors:   Float number shows L2 norm of error vector (||x-x_true||_2) 
    '''
    
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x_true = np.asarray(x_true, dtype=float)
    x = np.asarray(x0, dtype=float)
    
    N = len(b)
    
    # Gauss-Seidel iteration
    while True:
        x_old = x.copy()
        
        # Update each component using forward substitution
        for i in range(N):
            # x_i = (b_i - sum of lower triangle part - sum of upper triangle part) / A_ii
            sum_lower = np.dot(A[i, :i], x[:i])  # uses updated values
            sum_upper = np.dot(A[i, i+1:], x_old[i+1:])  # uses old values
            
            x[i] = (b[i] - sum_lower - sum_upper) / A[i, i]
        
        # Check convergence: ||x_k - x_{k-1}||_2 < eps
        increment = np.linalg.norm(x - x_old)
        if increment < eps:
            break
    
    # Compute residual: ||Ax - b||_2
    residual = np.linalg.norm(A @ x - b)
    
    # Compute error: ||x - x_true||_2
    error = np.linalg.norm(x - x_true)
    
    return residual, error


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('3.1', 3)
target = targets[0]

n = 7
h = 1/(n-1)
diagonal = [2/h for i in range(n)]
diagonal_up = [-1/h for i in range(n-1)]
diagonal_down = [-1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
A[:, 0] = 0
A[0, :] = 0
A[0, 0] = 1/h
A[:, -1] = 0
A[-1, :] = 0
A[7-1, 7-1] = 1/h
b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
x_true = np.linalg.solve(A, b)
eps = 10e-5
x0 = np.zeros(n)
assert np.allclose(GS(A, b, eps, x_true, x0), target)
target = targets[1]

n = 7
h = 1/(n-1)
diagonal = [2/h for i in range(n)]
diagonal_up = [-0.5/h for i in range(n-2)]
diagonal_down = [-0.5/h for i in range(n-2)]
A = np.diag(diagonal) + np.diag(diagonal_up, 2) + np.diag(diagonal_down, -2)
b = np.array([0.5,0.1,0.5,0.1,0.5,0.1,0.5])
x_true = np.linalg.solve(A, b)
eps = 10e-5
x0 = np.zeros(n)
assert np.allclose(GS(A, b, eps, x_true, x0), target)
target = targets[2]

n = 7
h = 1/(n-1)
diagonal = [2/h for i in range(n)]
diagonal_2up = [-0.5/h for i in range(n-2)]
diagonal_2down = [-0.5/h for i in range(n-2)]
diagonal_1up = [-0.3/h for i in range(n-1)]
diagonal_1down = [-0.5/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_2up, 2) + np.diag(diagonal_2down, -2) + np.diag(diagonal_1up, 1) + np.diag(diagonal_1down, -1)
b = np.array([0.5,0.1,0.5,0.1,-0.1,-0.5,-0.5])
x_true = np.linalg.solve(A, b)
eps = 10e-5
x0 = np.zeros(n)
assert np.allclose(GS(A, b, eps, x_true, x0), target)
