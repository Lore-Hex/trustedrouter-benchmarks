import numpy as np



# Background: The Incomplete Cholesky (IC) factorization is a sparse matrix decomposition technique
# that computes an approximate lower triangular matrix L such that A ≈ LL^T. Unlike the standard
# Cholesky factorization which is dense, the IC factorization preserves the sparsity pattern of
# the original matrix, making it useful for preconditioning large sparse systems. The algorithm
# iterates through each row i, computes the lower triangular elements by:
# 1. For each column j < i: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k<j)) / L[j,j]
# 2. Only non-zero entries in the original matrix A are computed (sparsity preservation)
# 3. The diagonal: L[i,i] = sqrt(max(0, A[i,i] - sum(L[i,k]^2 for k<i)))
# This creates a lower triangular matrix that can serve as a preconditioner for iterative solvers.

def ichol(A):
    '''Inputs:
    A : Matrix, 2d array M * M
    Outputs:
    A : Matrix, 2d array M * M (modified in-place to store L, lower triangular)
    '''
    M = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    
    # Process each row
    for i in range(M):
        # Process each column in the lower triangular part
        for j in range(i):
            # Only compute if A[i,j] is non-zero (sparsity preservation)
            if A[i, j] != 0:
                # L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k<j)) / L[j,j]
                sum_val = 0.0
                for k in range(j):
                    sum_val += L[i, k] * L[j, k]
                
                if L[j, j] != 0:
                    L[i, j] = (A[i, j] - sum_val) / L[j, j]
                else:
                    L[i, j] = 0.0
        
        # Compute diagonal element
        sum_diag = 0.0
        for k in range(i):
            sum_diag += L[i, k] ** 2
        
        diag_val = A[i, i] - sum_diag
        L[i, i] = np.sqrt(max(0.0, diag_val))
    
    return L


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('4.1', 3)
target = targets[0]

n = 7
h = 1.0/n
diagonal = [2/h for i in range(n)]
diagonal_up = [-1/h for i in range(n-1)]
diagonal_down = [-1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
assert np.allclose(ichol(A), target)
target = targets[1]

n = 7
h = 1.0/n
diagonal = [1/h for i in range(n)]
diagonal_up = [-0.1/h for i in range(n-1)]
diagonal_down = [-0.1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
assert np.allclose(ichol(A), target)
target = targets[2]

n = 7
h = 1.0/n
diagonal = [2/h for i in range(n)]
diagonal_up = [-0.3/h for i in range(n-2)]
diagonal_down = [-0.3/h for i in range(n-2)]
A = np.diag(diagonal) + np.diag(diagonal_up, 2) + np.diag(diagonal_down, -2)
assert np.allclose(ichol(A), target)
