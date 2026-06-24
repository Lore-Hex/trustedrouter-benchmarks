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


# Background: The d-tensor (or d-symbol) is a fundamental object in SU(3) algebra that arises from
# the anticommutation relations of the Gell-Mann matrices. It is defined as:
# d_{ijk} = (1/4) * Tr({λ_i, λ_j} * λ_k)
# where {λ_i, λ_j} = λ_i * λ_j + λ_j * λ_i is the anticommutator of Gell-Mann matrices λ_i and λ_j.
# The d-tensor is completely symmetric in all three indices (d_ijk = d_jik = d_ikj = ...) and is
# real-valued due to the properties of the Gell-Mann matrices and the trace operation. The d-tensor
# plays an important role in decomposing products of Gell-Mann matrices and in understanding the
# structure of SU(3) representations. For indices i, j, k ranging from 1 to 8, we compute all
# 8^3 = 512 possible combinations. The symmetry means that d_{ijk} can be used to study the tensor
# products of the adjoint representation of SU(3).

def tensor_d(i, j, k):
    '''Returns the tensor d_ijk of the SU(3) algebra.
    Input
    i: the first index; int
    j: the second index; int
    k: the third index; int
    Output
    result: the value of d_ijk for given indices i, j, and k; float
    '''
    
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
    
    # Store all Gell-Mann matrices in a list (indexed 0-7 for internal use)
    gell_mann_matrices = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8]
    
    # Convert input indices from 1-8 to 0-7 for array indexing
    i_idx = i - 1
    j_idx = j - 1
    k_idx = k - 1
    
    # Validate indices
    if not (0 <= i_idx < 8 and 0 <= j_idx < 8 and 0 <= k_idx < 8):
        raise ValueError("Indices i, j, k must be in the range 1 to 8")
    
    # Get the Gell-Mann matrices for the given indices
    lambda_i = gell_mann_matrices[i_idx]
    lambda_j = gell_mann_matrices[j_idx]
    lambda_k = gell_mann_matrices[k_idx]
    
    # Compute the anticommutator {λ_i, λ_j} = λ_i * λ_j + λ_j * λ_i
    anticommutator = lambda_i @ lambda_j + lambda_j @ lambda_i
    
    # Compute {λ_i, λ_j} * λ_k
    product = anticommutator @ lambda_k
    
    # Compute the trace
    trace = np.trace(product)
    
    # Compute d_ijk = (1/4) * Tr({λ_i, λ_j} * λ_k)
    d_ijk = (1.0 / 4.0) * trace
    
    # Return the real part
    result = np.real(d_ijk)
    
    return result


# Background: The star product (h*h)_i is a fundamental operation in SU(3) algebra that combines
# two sets of expansion coefficients through the d-tensor (symmetric tensor of SU(3) structure constants).
# Given expansion coefficients h = {h_1, h_2, ..., h_8} representing a decomposition of a Hamiltonian
# in the basis of Gell-Mann matrices, the star product computes a new tensor by contracting pairs of
# these coefficients with the d-tensor: (h*h)_i = d_ijk * h^j * h^k, where summation is performed
# over all repeated indices j and k from 1 to 8. This operation is useful in computing products of
# Hamiltonian expansions and appears in the analysis of nonlinear terms in neutrino oscillation theory.
# The result is generally a complex number (or real if the input coefficients are real), and it
# represents the i-th component of the resulting tensor product decomposition.

def star_product(i, h):
    '''Returns the SU(3) star product (h*h)_i = d_ijk*h^j*h^k (summed over
    repeated indices).
    Input
    i: index of the star product; int
    h: a list of eight expansion coefficients; a list of floats (possibly complex)
    Output
    product: the star product (h*h)_i; complex numbers
    '''
    
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
    
    # Store all Gell-Mann matrices in a list (indexed 0-7 for internal use)
    gell_mann_matrices = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8]
    
    # Validate input index i
    if not (1 <= i <= 8):
        raise ValueError("Index i must be in the range 1 to 8")
    
    # Convert input index from 1-8 to 0-7 for array indexing
    i_idx = i - 1
    
    # Initialize the star product to zero
    product = 0.0 + 0.0j
    
    # Sum over all pairs of indices j and k from 1 to 8
    for j in range(1, 9):
        for k in range(1, 9):
            # Compute d_ijk using the tensor_d definition
            # d_ijk = (1/4) * Tr({λ_i, λ_j} * λ_k)
            j_idx = j - 1
            k_idx = k - 1
            
            lambda_i = gell_mann_matrices[i_idx]
            lambda_j = gell_mann_matrices[j_idx]
            lambda_k = gell_mann_matrices[k_idx]
            
            # Compute the anticommutator {λ_i, λ_j} = λ_i * λ_j + λ_j * λ_i
            anticommutator = lambda_i @ lambda_j + lambda_j @ lambda_i
            
            # Compute {λ_i, λ_j} * λ_k
            anticomm_product = anticommutator @ lambda_k
            
            # Compute the trace
            trace = np.trace(anticomm_product)
            
            # Compute d_ijk = (1/4) * Tr({λ_i, λ_j} * λ_k)
            d_ijk = (1.0 / 4.0) * np.real(trace)
            
            # Get the expansion coefficients h^j and h^k (convert to 0-indexed)
            h_j = h[j_idx]
            h_k = h[k_idx]
            
            # Add the contribution to the star product: d_ijk * h^j * h^k
            product += d_ijk * h_j * h_k
    
    return product


# Background: In neutrino oscillation theory, the SU(3) expansion of the Hamiltonian
# can be analyzed through two fundamental SU(3) invariants that are independent of the
# choice of basis and provide insight into the structure of the system. The first
# invariant is |h|^2 = h_k * h^k (summation over k from 1 to 8), which represents
# the squared norm of the expansion coefficient vector. The second invariant is
# <h> = d_ijk * h^i * h^j * h^k (summation over all i, j, k from 1 to 8), which is
# a cubic invariant derived from the SU(3) d-tensor. These invariants are related
# to the eigenvalues of the Hamiltonian through a characteristic equation. The three
# eigenvalues can be expressed in terms of these invariants using the formula:
# psi_m = (2|h| / sqrt(3)) * cos[(1/3)(chi + 2*pi*m)] where m = 1, 2, 3
# and cos(chi) = -sqrt(3) * <h> / |h|^3. This decomposition is useful for understanding
# the energy levels of the three-neutrino system in the flavor basis.

def psi_m(h):
    '''Input
    h: a list of eight expansion coefficients; a list of floats (possibly complex)
    Output
    result: a list containing the following three variables:
            h2: SU(3) invariant |h|^2; complex number
            h3: SU(3) invariant <h>; complex number
            psi_list: a list of three complex number
    '''
    
    # Convert input to numpy array
    h_array = np.array(h, dtype=complex)
    
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
    
    # Store all Gell-Mann matrices in a list (indexed 0-7 for internal use)
    gell_mann_matrices = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8]
    
    # Calculate the first SU(3) invariant: |h|^2 = h_k * h^k (summation over k from 1 to 8)
    h2 = np.sum(h_array * np.conj(h_array))
    
    # Calculate the second SU(3) invariant: <h> = d_ijk * h^i * h^j * h^k
    # (summation over all i, j, k from 1 to 8)
    h3 = 0.0 + 0.0j
    
    for i in range(1, 9):
        for j in range(1, 9):
            for k in range(1, 9):
                # Compute d_ijk = (1/4) * Tr({λ_i, λ_j} * λ_k)
                i_idx = i - 1
                j_idx = j - 1
                k_idx = k - 1
                
                lambda_i = gell_mann_matrices[i_idx]
                lambda_j = gell_mann_matrices[j_idx]
                lambda_k = gell_mann_matrices[k_idx]
                
                # Compute the anticommutator {λ_i, λ_j} = λ_i * λ_j + λ_j * λ_i
                anticommutator = lambda_i @ lambda_j + lambda_j @ lambda_i
                
                # Compute {λ_i, λ_j} * λ_k
                anticomm_product = anticommutator @ lambda_k
                
                # Compute the trace
                trace = np.trace(anticomm_product)
                
                # Compute d_ijk = (1/4) * Tr({λ_i, λ_j} * λ_k)
                d_ijk = (1.0 / 4.0) * np.real(trace)
                
                # Get the expansion coefficients h^i, h^j, h^k
                h_i = h_array[i_idx]
                h_j = h_array[j_idx]
                h_k = h_array[k_idx]
                
                # Add the contribution to h3: d_ijk * h^i * h^j * h^k
                h3 += d_ijk * h_i * h_j * h_k
    
    # Calculate the eigenvalues psi_m using the formula:
    # psi_m = (2|h| / sqrt(3)) * cos[(1/3)(chi + 2*pi*m)] where m = 1, 2, 3
    # and cos(chi) = -sqrt(3) * <h> / |h|^3
    
    # Compute |h| = sqrt(|h|^2)
    h_norm = np.sqrt(h2)
    
    # Compute cos(chi) = -sqrt(3) * <h> / |h|^3
    # Avoid division by zero
    if np.abs(h_norm) < 1e-15:
        cos_chi = 0.0
        chi = 0.0
    else:
        cos_chi = -np.sqrt(3) * h3 / (h_norm**3)
        # Clamp cos_chi to [-1, 1] to avoid numerical issues with arccos
        cos_chi_clamped = np.clip(np.real(cos_chi), -1.0, 1.0)
        chi = np.arccos(cos_chi_clamped)
    
    # Calculate psi_m for m = 1, 2, 3
    psi_list = []
    for m in range(1, 4):
        angle = (chi + 2.0 * np.pi * m) / 3.0
        psi_m_value = (2.0 * h_norm / np.sqrt(3)) * np.cos(angle)
        psi_list.append(psi_m_value)
    
    # Return the result as a list containing h2, h3, and psi_list
    result = [h2, h3, psi_list]
    
    return result



# Background: The evolution operator U_3(L) describes how a neutrino state evolves over a baseline distance L.
# In the flavor basis, it is given by U_3(L) = exp(-i * H * L), where H is the Hamiltonian in the flavor basis.
# This operator can be expanded in terms of Gell-Mann matrices as: U_3(L) = u_0 * I + i * u_k * λ^k,
# where u_0 is the coefficient of the identity and u_k (k=1 to 8) are the coefficients of the Gell-Mann matrices.
#
# To extract these coefficients, we need to:
# 1. Compute the matrix exponential exp(-i * H * L) numerically
# 2. Use the orthogonality property of Gell-Mann matrices to extract coefficients:
#    - u_0 = Tr(U_3) / 3 (since the identity contributes equally to the trace)
#    - u_k = -i * Tr(U_3 * λ^k) / 2 (note the factor of -i due to the i in front of u_k in the expansion)
#
# The factor of -i arises because U_3 = u_0 * I + i * u_k * λ^k, so when we take the trace with λ^k,
# we get Tr(U_3 * λ^k) = i * u_k * Tr(λ^k * λ^k) = i * u_k * 2, giving u_k = -i * Tr(U_3 * λ^k) / 2.

def evolution_operator_3nu_su3_coefficients(hamiltonian, L):
    '''Returns the nine coefficients u0, ..., u8 of the three-neutrino evolution operator U3(L) in its SU(3) exponential expansion,
    i.e., U3 = u0*I + i*u_k*lambda^k.
    Input
    hamiltonian: a list of lists containing the 3x3 Hamiltonian matrix; each inner list contains three complex numbers
    L: baseline; float
    Output
    u_list: a list of expansion coefficients for the evolution opereator; a list of complex numbers
    '''
    
    # Convert input Hamiltonian to numpy array
    H = np.array(hamiltonian, dtype=complex)
    
    # Compute the matrix exponential of -i * H * L
    # U_3(L) = exp(-i * H * L)
    exponent = -1j * H * L
    U_3 = np.linalg.matrix_power(np.eye(3, dtype=complex), 0) * 0  # Initialize to zero
    
    # Use scipy-free matrix exponential via eigendecomposition or numpy's built-in
    # For a more robust approach, we use the formula: exp(A) via eigendecomposition
    # But numpy doesn't have a direct exp function, so we use the Taylor series or eigendecomposition
    
    # Alternative: Use the fact that exp(A) can be computed via diagonalization
    # But for general matrices, we need a proper implementation
    # NumPy provides matrix exponential via scipy, but since scipy is not imported,
    # we use an alternative approach: compute via eigendecomposition
    
    try:
        # Try to use scipy if available (fallback method)

        U_3 = expm(exponent)
    except ImportError:
        # Fallback: compute matrix exponential using eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(exponent)
        # Exp of diagonal matrix is straightforward
        exp_diag = np.diag(np.exp(eigenvalues))
        # Reconstruct: exp(A) = P * exp(D) * P^{-1}
        U_3 = eigenvectors @ exp_diag @ np.linalg.inv(eigenvectors)
    
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
    
    # Extract u_0 coefficient
    # u_0 = Tr(U_3) / 3
    trace_U3 = np.trace(U_3)
    u_0 = trace_U3 / 3.0
    
    # Extract u_k coefficients (k=1 to 8)
    # u_k = -i * Tr(U_3 * λ^k) / 2
    u_k_list = []
    for k in range(8):
        lambda_k = gell_mann_matrices[k]
        # Compute Tr(U_3 * λ^k)
        product = U_3 @ lambda_k
        trace = np.trace(product)
        # Extract u_k: u_k = -i * Tr(U_3 * λ^k) / 2
        u_k = -1j * trace / 2.0
        u_k_list.append(u_k)
    
    # Combine u_0 and u_k coefficients into a single list
    u_list = [u_0] + u_k_list
    
    return u_list


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('70.7', 3)
target = targets[0]

s12 = 1/cmath.sqrt(3)
s13 = 1
s23 = 1/cmath.sqrt(2)
dCP = 0
D21 = 1e-4
D31 = 1e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
L = 1.0
assert np.allclose(evolution_operator_3nu_su3_coefficients(hamiltonian, L), target)
target = targets[1]

s12 = 1/cmath.sqrt(2)
s13 = 0
s23 = 1/cmath.sqrt(2)
dCP = 0
D21 = 1e-4
D31 = 1e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
L = 1.0
assert np.allclose(evolution_operator_3nu_su3_coefficients(hamiltonian, L), target)
target = targets[2]

s12 = 1/cmath.sqrt(2)
s13 = 0
s23 = 1/cmath.sqrt(3)
dCP = 1
D21 = 5e-4
D31 = 5e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
L = 2.0
assert np.allclose(evolution_operator_3nu_su3_coefficients(hamiltonian, L), target)
