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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('70.1', 3)
target = targets[0]

s12 = 1/cmath.sqrt(2)
s13 = 0.2
s23 = 1/cmath.sqrt(2)
assert np.allclose(pmns_mixing_matrix(s12,s13,s23,0), target)
target = targets[1]

s12 = 1/cmath.sqrt(3)
s13 = 1
s23 = 1/cmath.sqrt(2)
assert np.allclose(pmns_mixing_matrix(s12,s13,s23,0), target)
target = targets[2]

s12 = 1/cmath.sqrt(3)
s13 = 0
s23 = 1/cmath.sqrt(4)
assert np.allclose(pmns_mixing_matrix(s12,s13,s23,1), target)
