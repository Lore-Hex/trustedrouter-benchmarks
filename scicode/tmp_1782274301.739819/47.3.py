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



# Background: The total energy of an entire molecular system is the sum of all pairwise Lennard-Jones interactions between distinct atoms. Since the Lennard-Jones potential is pairwise and symmetric (U_ij = U_ji), the total system energy is calculated by summing the potential energies for all unique pairs (i, j) where i < j. This avoids double-counting, as each pair interaction is counted only once. The total system energy represents the overall potential energy landscape of all atoms interacting with one another through van der Waals forces and repulsive interactions. Alternatively, this can be computed as half the sum of individual atom energies, where each atom's energy is its interaction with all other atoms, since summing all individual atom energies counts each pairwise interaction twice (once for atom i interacting with j, and once for j interacting with i).

def U_ij(r, sigma, epsilon):
    '''Lennard Jones Potential between pair of atoms with distance r'''
    ratio = sigma / r
    U = 4 * epsilon * (ratio**12 - ratio**6)
    return U

def U_system(positions, sigma, epsilon):
    '''Total energy of entire system
    Inputs:
    positions: all atom positions, 2D array of floats with shape (N,3), where N is the number of atoms, 3 is for x,y,z coordinate
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    Outputs:
    U: Aggregated energy of entire system, float
    '''
    
    N = len(positions)
    U_total = 0.0
    
    # Iterate over all unique pairs (i, j) where i < j
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate distance between atom i and atom j
            r_ij = np.linalg.norm(positions[i] - positions[j])
            
            # Add the Lennard-Jones potential contribution to total energy
            U_total += U_ij(r_ij, sigma, epsilon)
    
    return U_total


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('47.3', 3)
target = targets[0]

positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
sigma = 1.0
epsilon = 1.0
assert np.allclose(U_system(positions,sigma,epsilon), target)
target = targets[1]

positions = np.array([[0.1, 1.5, 0.3], [0.8, 0.0, 0.0], [1.5, 1.5, 0.0], [2.5, 2.5, 0.0]])
sigma = 1.0
epsilon = 2.0
assert np.allclose(U_system(positions,sigma,epsilon), target)
target = targets[2]

positions = np.array([[0.0, 0.0, 0.0], [0.0, 7.0, 0.0], [10, 0.0, 0.0]])
sigma = 1.0
epsilon = 1.0
assert np.allclose(U_system(positions,sigma,epsilon), target)
