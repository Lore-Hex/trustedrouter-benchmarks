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



# Background: The Metropolis-Hastings Algorithm is a Markov Chain Monte Carlo (MCMC) method for sampling from 
# a probability distribution, particularly useful in statistical mechanics. In molecular dynamics at temperature T, 
# the system samples configurations according to the Boltzmann distribution: P(state) ∝ exp(-U/k_B*T), where U is 
# the potential energy, k_B is Boltzmann's constant, and T is temperature. In reduced units (k_B=1), this becomes 
# P(state) ∝ exp(-U/T).
#
# The Metropolis-Hastings algorithm works as follows:
# 1. Start with an initial configuration (positions) and compute its energy U_old.
# 2. For each MC step:
#    a. Generate a trial configuration by displacing each atom randomly with a Gaussian distribution of width dispSize.
#    b. Compute the trial energy U_new.
#    c. Calculate the acceptance probability: A = min(1, exp(-(U_new - U_old)/T))
#    d. Accept the trial move with probability A; otherwise, reject and keep the old configuration.
#    e. Record the energy of the current configuration.
# 3. Return a trace of energies for all MC steps.
#
# The Gaussian trial move displaces each atom independently by a random amount drawn from N(0, dispSize^2),
# maintaining the system's dynamics at thermal equilibrium at temperature T.

def MC(sigma, epsilon, T, init_positions, MC_steps, dispSize):
    '''Markov Chain Monte Carlo simulation to generate samples energy of system of atoms interacting through Lennard Jones potential
    at temperature T, using Metropolis-Hasting Algorithm with Gaussian trial move
    Inputs:
    sigma: the distance at which Lennard Jones potential reaches zero, float,
    epsilon: potential well depth of Lennard Jones potential, float 
    T: Temperature in reduced unit, float
    init_positions: initial position of all atoms, 2D numpy array of float with shape (N, 3) where N is number of atoms, 3 is for x,y,z coordinate
    MC_steps: Number of MC steps to perform, int
    dispSize: Size of displacement in Gaussian trial move, float
    Outputs:
    E_trace: Samples of energy obtained by MCMC sampling start with initial energy, list of floats with length MC_steps+1 
    '''
    
    # Initialize positions and compute initial energy
    positions = np.copy(init_positions).astype(float)
    U_current = U_system(positions, sigma, epsilon)
    
    # Initialize energy trace with initial energy
    E_trace = [U_current]
    
    # Perform MC steps
    for step in range(MC_steps):
        # Generate trial positions by Gaussian displacement
        displacement = np.random.normal(0, dispSize, positions.shape)
        positions_trial = positions + displacement
        
        # Compute trial energy
        U_trial = U_system(positions_trial, sigma, epsilon)
        
        # Calculate energy difference
        dU = U_trial - U_current
        
        # Metropolis-Hastings acceptance criterion
        if dU < 0:
            # Accept if energy decreases
            accept = True
        else:
            # Accept with probability exp(-dU/T)
            acceptance_prob = np.exp(-dU / T)
            accept = np.random.random() < acceptance_prob
        
        # Update configuration and energy if accepted
        if accept:
            positions = positions_trial
            U_current = U_trial
        
        # Record the current energy
        E_trace.append(U_current)
    
    return np.array(E_trace)


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('47.4', 3)
target = targets[0]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
# fix random seed for reproducibility
np.random.seed(1024)
sigma = 1
epsilon = 3
T = 1
MC_steps = 1000
N = 32
init_positions = initialize_fcc(N)
dispSize = 1.0
E_trace = MC(sigma,epsilon,T,init_positions,MC_steps,dispSize)
assert np.allclose(E_trace, target)
target = targets[1]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
sigma = 2
epsilon = 3
T = 5
MC_steps = 10
N = 4
init_positions = initialize_fcc(N)
dispSize = 2.0
E_trace = MC(sigma,epsilon,T,init_positions,MC_steps, dispSize)
assert np.allclose(E_trace, target)
target = targets[2]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
sigma = 3
epsilon = 10
T = 7
MC_steps = 10
N = 64
init_positions = initialize_fcc(N)
dispSize = 0.1
E_trace = MC(sigma,epsilon,T,init_positions,MC_steps, dispSize)
assert np.allclose(E_trace, target)
