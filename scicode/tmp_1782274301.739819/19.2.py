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



# Background: The n-tangle (also called the n-way entanglement measure) of a pure n-qubit state is a generalization
# of the concurrence and 3-tangle to arbitrary n-qubit systems. For an n-qubit pure state |ψ⟩, the n-tangle 
# quantifies the multipartite entanglement present in the state.
#
# For a pure n-qubit state |ψ⟩, the n-tangle is computed as follows:
# 1. Construct the "partial transpose" or density matrix by taking outer product: ρ = |ψ⟩⟨ψ|
# 2. For the n-tangle of an n-qubit system (n even), we need to compute the reduced density matrices
#    by tracing out different subsets of qubits and measuring entanglement across bipartitions.
# 3. The n-tangle can be computed using the formula involving the reduced density matrices and their
#    eigenvalues. For an even n, one common approach uses the "multipartite concurrence" concept.
# 4. The n-tangle of a pure state |ψ⟩ is defined as:
#    τ_n = (2^(n-1) - 1) * min(C(ρ_i)) for all single-qubit reduced density matrices,
#    or more generally as a measure of genuinely multipartite entanglement.
# 5. For practical computation of pure state n-tangle, we use the formula based on the
#    "residual entanglement" which can be extracted from the partial transpose and its eigenvalues.
# 6. The most direct definition: for a pure n-qubit state, compute all bipartitions and their
#    concurrence-like measures, then combine them appropriately for the n-tangle.
#
# For even n, a standard approach is to use the absolute value of the determinant of the
# reduced density matrix under specific partial transpose operations, or to compute it via
# the state's Schmidt decomposition across different bipartitions.

def n_tangle(psi):
    '''Returns the n_tangle of pure state psi
    Input:
    psi: 1d array of floats, the vector representation of the state
    Output:
    tangle: float, the n-tangle of psi
    '''
    
    psi = np.array(psi, dtype=complex)
    
    # Determine n from the state vector dimension (dim = 2^n)
    dim = len(psi)
    n = int(np.log2(dim))
    
    if 2**n != dim:
        raise ValueError("State vector dimension must be a power of 2")
    
    if n % 2 != 0:
        raise ValueError("n must be even for n-tangle computation")
    
    # Compute the density matrix of the pure state
    rho = np.outer(psi, np.conj(psi))
    
    # For even n-qubit system, compute n-tangle using multipartite entanglement
    # One standard approach: use the concurrence across (n/2)-(n/2) bipartition
    # This requires reshaping and partial tracing
    
    # Reshape state vector to tensor form for bipartition analysis
    # |ψ⟩ as a (2^(n/2)) × (2^(n/2)) matrix (bipartition into two halves)
    half_dim = 2 ** (n // 2)
    psi_mat = psi.reshape((half_dim, half_dim))
    
    # Compute singular values of the reshaped state matrix (Schmidt decomposition)
    U, S, Vh = np.linalg.svd(psi_mat)
    
    # Compute concurrence-like measure from Schmidt coefficients
    # For the n-tangle of a pure state under bipartition, use square of Schmidt coefficients
    lambda_sq = S ** 2
    
    # The n-tangle can be computed as a product measure of entanglement across bipartitions
    # For even n, compute as: τ_n = 2^(n-1) * |det(partial_transpose(rho))|
    # Or use the approach based on all bipartitions
    
    # Alternative direct formula for pure states:
    # Compute the reduced density matrices for single qubits and use their concurrence
    # For a simpler and more direct approach for pure states:
    
    # The n-tangle for pure even-n states can be computed as:
    # τ_n = 2^(n-1) * |λ_1 * λ_2 * ... * λ_(n/2)| where λ_i are Schmidt coefficients
    
    tangle = (2 ** (n - 1)) * np.prod(lambda_sq)
    
    return float(np.real(tangle))


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('19.2', 4)
target = targets[0]

MaxEnt = np.array([1,0,0,1])/np.sqrt(2)
assert np.allclose(n_tangle(MaxEnt), target)
target = targets[1]

GHZ = np.zeros(16)
GHZ[0] = 1/np.sqrt(2)
GHZ[15] = 1/np.sqrt(2)
assert np.allclose(n_tangle(GHZ), target)
target = targets[2]

W = np.zeros(16)
W[1] = 1/2
W[2] = 1/2
W[4] = 1/2
W[8] = 1/2
assert np.allclose(n_tangle(W), target)
target = targets[3]

product_state = np.kron(np.array([0,1]),np.array([0.8,0.6]))
assert np.allclose(n_tangle(product_state), target)
