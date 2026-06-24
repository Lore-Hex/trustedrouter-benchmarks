import numpy as np
from scipy.special import erfc



# Background: 
# The Ewald summation is a technique for calculating long-range Coulombic interactions in periodic systems.
# The alpha parameter controls the division of interactions into short-range (real space) and long-range 
# (reciprocal space) components. A Gaussian charge distribution with width parameter alpha is used to 
# partition the Coulomb potential into two complementary parts that converge faster than the original sum.
# 
# The alpha value is typically determined from the reciprocal lattice vectors to ensure an appropriate
# balance between real and reciprocal space summation. The standard approach is to use the magnitude of
# the smallest reciprocal lattice vector (or a related measure) as the basis for alpha. The formula is:
# 
# alpha = alpha_scaling / (2 * pi * r_min)
# 
# where r_min is related to the shortest distance in reciprocal space. A common choice is to compute
# the norm of the reciprocal lattice vectors and use their minimum or geometric mean as the scale.
# Alternatively, alpha can be derived from the volume of the reciprocal lattice cell or the 
# characteristic length scale. For robustness, alpha is often set as:
#
# alpha = alpha_scaling * sqrt(det(G)) / (2 * pi)
# where G is the reciprocal metric tensor (recvec @ recvec.T), and sqrt(det(G)) relates to the
# reciprocal space volume.

def get_alpha(recvec, alpha_scaling=5):
    '''Calculate the alpha value for the Ewald summation, scaled by a specified factor.
    Parameters:
        recvec (np.ndarray): A 3x3 array representing the reciprocal lattice vectors.
        alpha_scaling (float): A scaling factor applied to the alpha value. Default is 5.
    Returns:
        float: The calculated alpha value.
    '''
    
    # Compute the reciprocal metric tensor G = recvec @ recvec.T
    G = recvec @ recvec.T
    
    # Calculate the determinant of G
    det_G = np.linalg.det(G)
    
    # Ensure det_G is positive (it should be for a valid reciprocal lattice)
    det_G = np.abs(det_G)
    
    # Calculate alpha using the formula: alpha = alpha_scaling * sqrt(det(G)) / (2 * pi)
    alpha = alpha_scaling * np.sqrt(det_G) / (2 * np.pi)
    
    return alpha


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('10.1', 4)
target = targets[0]

ref1 = -1.74756
EX1 = {
    'latvec': np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
        ]),
    'atom_charges': np.array([1]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0]
        ]),
    'configs': np.array([
        [1.0, 1.0, 1.0]
    ]),
}
assert np.allclose(get_alpha(np.linalg.inv(EX1['latvec']).T), target)
target = targets[1]

ref2 = -6.99024
EX2 = {
    'latvec': np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
        ]),
    'atom_charges': np.array([1, 1, 1, 1]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
        ]),
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
        ])    
}
assert np.allclose(get_alpha(np.linalg.inv(EX2['latvec']).T), target)
target = targets[2]

ref3 = -5.03879
L = 4 / 3**0.5
EX3 = {
    'latvec': (np.ones((3, 3)) - np.eye(3)) * L / 2,
    'atom_charges': np.array([2]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0]
        ]),
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],        
        ]) * L/4
    }
assert np.allclose(get_alpha(np.linalg.inv(EX3['latvec']).T), target)
target = targets[3]

ref4 = -20.15516
EX4 = {
    'latvec': np.eye(3, 3) * L,
    'atom_charges': np.array([2, 2, 2, 2]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0] 
    ]) * L/2,
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 3.0],
        [1.0, 3.0, 1.0],
        [1.0, 3.0, 3.0],
        [3.0, 1.0, 1.0],
        [3.0, 1.0, 3.0],
        [3.0, 3.0, 1.0],
        [3.0, 3.0, 3.0]        
    ]) * L/4
}
assert np.allclose(get_alpha(np.linalg.inv(EX4['latvec']).T), target)
