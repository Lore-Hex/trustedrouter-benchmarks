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



# Background: In a molecular system with multiple atoms, the total energy of a single atom is determined by its pairwise interactions with all other atoms. The total energy of atom i is the sum of Lennard-Jones potentials between atom i and every other atom j in the system. Since the Lennard-Jones potential is pairwise and symmetric (U_ij = U_ji), we sum the potential energy contributions from all j != i. This aggregated energy reflects how much the presence of all other atoms contributes to the potential energy landscape experienced by atom i.

def U_i(r_i, i, positions, sigma, epsilon):
    '''Total energy on a single atom
    Inputs:
    r_i: atom position, 1d array of floats with x,y,z coordinate
    i: atom index, int
    positions: all atom positions, 2D array of floats with shape (N,3), where N is the number of atoms, 3 is x,y,z coordinate
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    Outputs:
    U_i: Aggregated energy on particle i, float
    '''
    
    U_i_total = 0.0
    
    # Iterate over all other atoms
    for j in range(len(positions)):
        # Skip self-interaction
        if j == i:
            continue
        
        # Calculate distance between atom i and atom j
        r_ij = np.linalg.norm(positions[i] - positions[j])
        
        # Add the Lennard-Jones potential contribution from atom j to atom i
        U_i_total += U_ij(r_ij, sigma, epsilon)
    
    return U_i_total


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('47.2', 3)
target = targets[0]

r_i = np.array([0.0, 0.0, 0.0])
i = 0
positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
sigma = 1.0
epsilon = 1.0
assert np.allclose(U_i(r_i,i,positions,sigma,epsilon), target)
target = targets[1]

r_i = np.array([0.5, 0.7, 0.3])
i = 1
positions = np.array([[0.1, 1.5, 0.3], [0.8, 0.0, 0.0], [1.5, 1.5, 0.0], [2.5, 2.5, 0.0]])
sigma = 1.0
epsilon = 2.0
assert np.allclose(U_i(r_i,i,positions,sigma,epsilon), target)
target = targets[2]

r_i = np.array([0.0, 2.0, 0.0])
i = 2
positions = np.array([[0.0, 0.0, 0.0], [0.0, 7.0, 0.0], [10, 0.0, 0.0]])
sigma = 1.0
epsilon = 1.0
assert np.allclose(U_i(r_i,i,positions,sigma,epsilon), target)
