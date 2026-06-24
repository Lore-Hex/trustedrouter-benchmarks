import numpy as np
from scipy.linalg import sqrtm



# Background: The tensor product (Kronecker product) of matrices is a fundamental operation in linear algebra and quantum mechanics. 
# For matrices A (m×n) and B (p×q), the Kronecker product A ⊗ B is an (mp)×(nq) matrix defined as:
# A ⊗ B = [[a_11*B, a_12*B, ..., a_1n*B],
#           [a_21*B, a_22*B, ..., a_2n*B],
#           [...],
#           [a_m1*B, a_m2*B, ..., a_mn*B]]
# This operation is associative but not commutative, allowing computation of sequential tensor products.
# For multiple matrices/vectors, the tensor product is computed iteratively: A ⊗ B ⊗ C = (A ⊗ B) ⊗ C.
# NumPy's kron function efficiently computes the Kronecker product of two arrays.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of arrays, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices
    '''
    
    if len(args) == 0:
        return np.array([])
    
    if len(args) == 1:
        return args[0]
    
    # Start with the first matrix
    M = np.array(args[0])
    
    # Iteratively compute the tensor product with each subsequent matrix
    for matrix in args[1:]:
        M = np.kron(M, np.array(matrix))
    
    return M


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('19.1', 3)
target = targets[0]

assert np.allclose(tensor([0,1],[0,1]), target)
target = targets[1]

assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
target = targets[2]

assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
