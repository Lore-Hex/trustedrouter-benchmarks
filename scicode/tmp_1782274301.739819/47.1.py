import numpy as np
from scipy.spatial.distance import pdist



# Background: The Lennard-Jones potential is a mathematical model that describes the interaction energy between a pair of neutral atoms or molecules. It is widely used in molecular dynamics simulations and computational chemistry.
# The potential has two main components:
# 1. Repulsive term (1/r^12): Dominates at short distances and represents electron cloud overlap and quantum mechanical repulsion.
# 2. Attractive term (1/r^6): Dominates at larger distances and represents van der Waals forces (London dispersion forces).
# 
# The Lennard-Jones potential formula is:
# U(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
# 
# Where:
# - r: the distance between two atoms
# - sigma: the distance at which the potential equals zero (the effective diameter of the atoms)
# - epsilon: the depth of the potential well (the maximum attractive energy)
# 
# The potential reaches its minimum at r = 2^(1/6) * sigma ≈ 1.122 * sigma, with a minimum value of -epsilon.
# At r = sigma, the potential is exactly zero by design.
# For r < sigma, the potential is positive (repulsive).
# For r > sigma, the potential is negative (attractive) until it approaches zero asymptotically.

def U_ij(r, sigma, epsilon):
    '''Lennard Jones Potential between pair of atoms with distance r
    Inputs:
    r: distance, float
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float
    Outputs:
    U: Potential Energy, float
    '''
    
    # Calculate the Lennard-Jones potential using the standard formula
    # U(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    
    ratio = sigma / r
    U = 4 * epsilon * (ratio**12 - ratio**6)
    
    return U


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('47.1', 3)
target = targets[0]

r = 0.5
sigma = 1
epsilon = 3
assert np.allclose(U_ij(r,sigma,epsilon), target)
target = targets[1]

r = 1.1
sigma = 2
epsilon = 1
assert np.allclose(U_ij(r,sigma,epsilon), target)
target = targets[2]

r = 5.0
sigma = 5
epsilon = 3
assert np.allclose(U_ij(r,sigma,epsilon), target)
