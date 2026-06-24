import numpy as np
import cmath 

# Background: The PMNS (Pontecorvo-Maki-Nakagawa-Sakata) matrix is the fundamental 3x3 unitary matrix
# in neutrino physics that describes the mixing of three neutrino mass eigenstates (nu_1, nu_2, nu_3)
# into three flavor eigenstates (nu_e, nu_mu, nu_tau). The matrix is parameterized by three Euler
# rotation angles (theta_12, theta_23, theta_13) and one CP-violation phase (delta_CP). The standard
# parameterization expresses the PMNS matrix as a product of three rotation matrices:
# U_PMNS = R_23(theta_23, 0) * R_13(theta_13, delta_CP) * R_12(theta_12, 0)
# where R_ij represents a rotation in the ij-plane, with the CP phase appearing in the R_13 rotation.
# Each element can be complex due to the CP-violation phase. The matrix must be unitary (U * U† = I).

def pmns_mixing_matrix(s12, s23, s13, dCP):
    '''Returns the 3x3 PMNS mixing matrix.
    Computes and returns the 3x3 complex PMNS mixing matrix
    parameterized by three rotation angles: theta_12, theta_23, theta_13,
    and one CP-violation phase, delta_CP.
    Input
    s12 : Sin(theta_12); float
    s23 : Sin(theta_23); float
    s13 : Sin(theta_13); float
    dCP : delta_CP in radians; float
    Output
    pmns: numpy array of complex numbers with shape (3,3) containing the 3x3 PMNS mixing matrix
    '''
    
    # Compute cosines from sines using sin^2 + cos^2 = 1
    c12 = np.sqrt(1 - s12**2)
    c23 = np.sqrt(1 - s23**2)
    c13 = np.sqrt(1 - s13**2)
    
    # Compute the CP-violation phase factor
    exp_i_dcp = cmath.exp(1j * dCP)
    
    # Construct the PMNS matrix as a product of three rotation matrices
    # U_PMNS = R_23(theta_23) * R_13(theta_13, delta_CP) * R_12(theta_12)
    
    # R_12 rotation matrix (in the 12-plane, no CP phase)
    R12 = np.array([
        [c12, s12, 0],
        [-s12, c12, 0],
        [0, 0, 1]
    ], dtype=complex)
    
    # R_13 rotation matrix (in the 13-plane, with CP phase)
    R13 = np.array([
        [c13, 0, s13 * np.conj(exp_i_dcp)],
        [0, 1, 0],
        [-s13 * exp_i_dcp, 0, c13]
    ], dtype=complex)
    
    # R_23 rotation matrix (in the 23-plane, no CP phase)
    R23 = np.array([
        [1, 0, 0],
        [0, c23, s23],
        [0, -s23, c23]
    ], dtype=complex)
    
    # Compute the product: R_23 * R_13 * R_12
    pmns = R23 @ R13 @ R12
    
    return pmns


# Background: In the three-neutrino oscillation problem, neutrinos propagate through space as
# superpositions of mass eigenstates. The time evolution of this system is governed by the 
# Schrödinger equation, where the Hamiltonian in the flavor basis (nu_e, nu_mu, nu_tau) is
# given by H = U_PMNS * diag(m1^2, m2^2, m3^2) * U_PMNS†, where U_PMNS is the mixing matrix
# and m_i are the neutrino masses. In practice, we work with mass-squared differences rather 
# than absolute masses. The Hamiltonian can be expressed as:
# H = (1/2E) * U_PMNS * diag(0, Delta_m21^2, Delta_m31^2) * U_PMNS†
# where Delta_m21^2 = m2^2 - m1^2 and Delta_m31^2 = m3^2 - m1^2 are the mass-squared 
# differences (with m1^2 subtracted from all terms, effectively setting it to zero).
# For the energy-independent Hamiltonian, we ignore the 1/(2E) factor and work with
# the mass matrix M^2 = diag(0, Delta_m21^2, Delta_m31^2) in the mass basis, then
# rotate it to the flavor basis using the PMNS matrix.

def hamiltonian_3nu(s12, s23, s13, dCP, D21, D31):
    '''Returns the energy-independent three-neutrino Hamiltonian for vacuum oscillations.
    Input
    s12 : Sin(theta_12); float
    s23 : Sin(theta_23); float
    s13 : Sin(theta_13); float
    dCP : delta_CP in radians; float
    D21 : Mass-squared difference Delta m^2_21; float
    D31 : Mass-squared difference Delta m^2_31; float
    Output
    hamiltonian: a list of lists containing the 3x3 Hamiltonian matrix; each inner list contains three complex numbers
    '''
    
    # Compute cosines from sines using sin^2 + cos^2 = 1
    c12 = np.sqrt(1 - s12**2)
    c23 = np.sqrt(1 - s23**2)
    c13 = np.sqrt(1 - s13**2)
    
    # Compute the CP-violation phase factor
    exp_i_dcp = cmath.exp(1j * dCP)
    
    # Construct the PMNS matrix as a product of three rotation matrices
    # U_PMNS = R_23(theta_23) * R_13(theta_13, delta_CP) * R_12(theta_12)
    
    # R_12 rotation matrix (in the 12-plane, no CP phase)
    R12 = np.array([
        [c12, s12, 0],
        [-s12, c12, 0],
        [0, 0, 1]
    ], dtype=complex)
    
    # R_13 rotation matrix (in the 13-plane, with CP phase)
    R13 = np.array([
        [c13, 0, s13 * np.conj(exp_i_dcp)],
        [0, 1, 0],
        [-s13 * exp_i_dcp, 0, c13]
    ], dtype=complex)
    
    # R_23 rotation matrix (in the 23-plane, no CP phase)
    R23 = np.array([
        [1, 0, 0],
        [0, c23, s23],
        [0, -s23, c23]
    ], dtype=complex)
    
    # Compute the PMNS matrix: U = R_23 * R_13 * R_12
    U_PMNS = R23 @ R13 @ R12
    
    # Mass-squared matrix in the mass basis (diagonal)
    # M^2 = diag(0, Delta_m21^2, Delta_m31^2)
    M_sq = np.array([
        [0, 0, 0],
        [0, D21, 0],
        [0, 0, D31]
    ], dtype=complex)
    
    # Transform to the flavor basis: H = U_PMNS * M^2 * U_PMNS†
    U_PMNS_dagger = np.conj(U_PMNS.T)
    hamiltonian_matrix = U_PMNS @ M_sq @ U_PMNS_dagger
    
    # Convert to list of lists for output
    hamiltonian = hamiltonian_matrix.tolist()
    
    return hamiltonian



# Background: The SU(3) group is the fundamental symmetry group describing strong interactions 
# and can also be used to decompose any 3x3 Hermitian matrix. The eight Gell-Mann matrices 
# (λ^k for k=1 to 8) form a complete basis for traceless Hermitian matrices in 3D, along with 
# the identity matrix. Any 3x3 complex Hamiltonian can be expressed as:
# H = h_0 * I + sum_{k=1}^{8} h_k * λ^k
# where I is the 3x3 identity matrix and λ^k are the Gell-Mann matrices. The coefficients h_k 
# are computed using the orthogonality property of Gell-Mann matrices:
# Tr(λ^j * λ^k) = 2 * δ_{jk} (Kronecker delta)
# The coefficients are extracted via: h_k = Tr(H * λ^k) / 2
# and h_0 = Tr(H) / 3 (since the Hamiltonian in the flavor basis for neutrinos is typically 
# traceless or nearly so, and the identity contributes equally to the trace).

def hamiltonian_3nu_su3_coefficients(hamiltonian):
    '''Returns the h_k of the SU(3)-expansion of the 3nu Hamiltonian.
    Input
    hamiltonian: a list of lists containing the 3x3 Hamiltonian matrix; each inner list contains three complex numbers
    Output
    hks: a list containing the h_k coefficients of the 3nu Hamiltonian in the expansion using the Gell-Mann matrices (k=1 to k=8); a list of floats (possibly complex)
    '''
    
    # Convert input list to numpy array
    H = np.array(hamiltonian, dtype=complex)
    
    # Define the 8 Gell-Mann matrices
    lambda_1 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=complex)
    
    lambda_2 = np.array([
        [0, -1j, 0],
        [1j, 0, 0],
        [0, 0, 0]
    ], dtype=complex)
    
    lambda_3 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=complex)
    
    lambda_4 = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ], dtype=complex)
    
    lambda_5 = np.array([
        [0, 0, -1j],
        [0, 0, 0],
        [1j, 0, 0]
    ], dtype=complex)
    
    lambda_6 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=complex)
    
    lambda_7 = np.array([
        [0, 0, 0],
        [0, 0, -1j],
        [0, 1j, 0]
    ], dtype=complex)
    
    lambda_8 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -2]
    ], dtype=complex) / np.sqrt(3)
    
    # Store all Gell-Mann matrices in a list
    gell_mann_matrices = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8]
    
    # Compute h_k coefficients using Tr(H * λ^k) / 2
    hks = []
    for k in range(8):
        lambda_k = gell_mann_matrices[k]
        # Compute the trace of H * λ^k
        product = H @ lambda_k
        trace = np.trace(product)
        # Extract the coefficient h_k
        h_k = trace / 2.0
        hks.append(h_k)
    
    return hks


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('70.3', 3)
target = targets[0]

s12 = 1/cmath.sqrt(3)
s13 = 1
s23 = 1/cmath.sqrt(2)
dCP = 0
D21 = 1e-4
D31 = 1e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
assert np.allclose(hamiltonian_3nu_su3_coefficients(hamiltonian), target)
target = targets[1]

s12 = 1/cmath.sqrt(2)
s13 = 0
s23 = 1/cmath.sqrt(2)
dCP = 0
D21 = 1e-4
D31 = 1e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
assert np.allclose(hamiltonian_3nu_su3_coefficients(hamiltonian), target)
target = targets[2]

s12 = 1/cmath.sqrt(2)
s13 = 0
s23 = 1/cmath.sqrt(3)
dCP = 1
D21 = 5e-4
D31 = 5e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
assert np.allclose(hamiltonian_3nu_su3_coefficients(hamiltonian), target)
