import numpy as np

# Background: The Lennard-Jones potential is a mathematical model describing the interaction between 
# neutral atoms or molecules. It has the form: V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6], where 
# epsilon is the potential well depth and sigma is the distance at which the potential reaches zero.
# The force between two particles is the negative gradient of the potential energy with respect to 
# displacement: F = -dV/dr. For the Lennard-Jones potential, the magnitude of the radial force is:
# F(r) = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]. Since force is a vector, we multiply this 
# radial force by the unit vector in the direction of r to get the 3D force vector. The force acts 
# along the line connecting the two particles, pushing them apart if repulsive (small r) or pulling 
# them together if attractive (larger r but still within the well).

def f_ij(sigma, epsilon, r):
    '''This function computes the force between two particles interacting through Lennard Jones Potantial
    Inputs:
    sigma: the distance at which potential reaches zero, float
    epsilon: potential well depth, float
    r: 3D displacement between two particles, 1d array of float with shape (3,)
    Outputs:
    f: force, 1d array of float with shape (3,)
    '''
    
    # Calculate the distance between particles
    r_magnitude = np.sqrt(np.sum(r**2))
    
    # Avoid division by zero
    if r_magnitude == 0:
        return np.zeros(3)
    
    # Calculate the ratio sigma/r
    sigma_over_r = sigma / r_magnitude
    
    # Calculate the radial force magnitude using Lennard-Jones force formula
    # F(r) = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]
    f_radial = 24 * epsilon / r_magnitude * (2 * (sigma_over_r**12) - (sigma_over_r**6))
    
    # Convert radial force to 3D vector force by multiplying by unit vector
    # Unit vector = r / r_magnitude
    f = f_radial * (r / r_magnitude)
    
    return f


# Background: In a many-body system, each atom experiences forces from all other atoms in the system.
# By Newton's third law, if atom i exerts a force on atom j, then atom j exerts an equal and opposite
# force on atom i. To find the net force on each atom, we must sum the pairwise forces from all other
# atoms. The aggregate force on atom i is: F_i = sum over all j≠i of F_ij, where F_ij is the force
# that atom j exerts on atom i (calculated from the displacement vector r_ij = r_i - r_j).
# Since the pairwise Lennard-Jones force function f_ij(r) returns the force on particle i due to
# particle j when given displacement r = r_i - r_j, we iterate over all pairs, compute their
# pairwise displacements, calculate the force from one particle on another, and accumulate these
# forces for each atom.

def aggregate_forces(sigma, epsilon, positions):
    '''This function aggregates the net forces on each atom for a system of atoms interacting through Lennard Jones Potential
    Inputs:
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    positions: 3D positions of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    Outputs:
    forces: net forces on each atom, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    '''
    
    N = positions.shape[0]
    forces = np.zeros((N, 3))
    
    # Helper function to calculate pairwise force (from previous step)
    def f_ij(sigma, epsilon, r):
        r_magnitude = np.sqrt(np.sum(r**2))
        if r_magnitude == 0:
            return np.zeros(3)
        sigma_over_r = sigma / r_magnitude
        f_radial = 24 * epsilon / r_magnitude * (2 * (sigma_over_r**12) - (sigma_over_r**6))
        f = f_radial * (r / r_magnitude)
        return f
    
    # Iterate over all pairs of atoms (i, j) where i != j
    for i in range(N):
        for j in range(N):
            if i != j:
                # Calculate displacement vector from atom j to atom i
                r_ij = positions[i] - positions[j]
                # Calculate force on atom i due to atom j
                force_ij = f_ij(sigma, epsilon, r_ij)
                # Accumulate force on atom i
                forces[i] += force_ij
    
    return forces


# Background: The Velocity Verlet algorithm is a numerical integration method used to solve 
# Newton's Second Law of Motion (F = ma) for a system of particles. It is a symplectic integrator
# that provides good energy conservation and stability for molecular dynamics simulations.
# The algorithm proceeds in three stages:
# 1. Update positions: x(t+dt) = x(t) + v(t)*dt + (1/2)*a(t)*dt^2
#    where a(t) = F(t)/m is the acceleration at the current time step
# 2. Calculate forces and accelerations at the new positions: a(t+dt) = F(t+dt)/m
# 3. Update velocities using the average acceleration:
#    v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt
# This two-step velocity update (half-step at beginning, half-step at end) ensures that the
# velocity is evaluated at the midpoint of the time interval, improving accuracy.
# The algorithm conserves energy well and is time-reversible, making it ideal for 
# long-duration molecular dynamics simulations with Lennard-Jones interactions.

def velocity_verlet(sigma, epsilon, positions, velocities, dt, m):
    '''This function runs Velocity Verlet algorithm to integrate the positions and velocities of atoms interacting through
    Lennard Jones Potential forward for one time step according to Newton's Second Law.
    Inputs:
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    positions: current positions of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    velocities: current velocities of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    dt: time step size, float
    m: mass, float
    Outputs:
    new_positions: new positions of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    new_velocities: new velocities of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    '''
    
    # Helper function to calculate pairwise force
    def f_ij(sigma, epsilon, r):
        r_magnitude = np.sqrt(np.sum(r**2))
        if r_magnitude == 0:
            return np.zeros(3)
        sigma_over_r = sigma / r_magnitude
        f_radial = 24 * epsilon / r_magnitude * (2 * (sigma_over_r**12) - (sigma_over_r**6))
        f = f_radial * (r / r_magnitude)
        return f
    
    # Helper function to aggregate forces
    def aggregate_forces(sigma, epsilon, positions):
        N = positions.shape[0]
        forces = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = positions[i] - positions[j]
                    force_ij = f_ij(sigma, epsilon, r_ij)
                    forces[i] += force_ij
        return forces
    
    # Stage 1: Calculate current accelerations
    current_forces = aggregate_forces(sigma, epsilon, positions)
    current_accelerations = current_forces / m
    
    # Stage 2: Update positions using x(t+dt) = x(t) + v(t)*dt + (1/2)*a(t)*dt^2
    new_positions = positions + velocities * dt + 0.5 * current_accelerations * (dt ** 2)
    
    # Stage 3: Calculate accelerations at new positions
    new_forces = aggregate_forces(sigma, epsilon, new_positions)
    new_accelerations = new_forces / m
    
    # Stage 4: Update velocities using v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt
    new_velocities = velocities + 0.5 * (current_accelerations + new_accelerations) * dt
    
    return new_positions, new_velocities



# Background: A Molecular Dynamics (MD) simulation evolves a system of particles forward in time by 
# repeatedly applying the equations of motion. The simulation starts with initial positions and velocities 
# of all atoms, then iteratively updates them using a numerical integration method (in this case, the 
# Velocity Verlet algorithm) for a specified number of time steps. Each time step:
# 1. Uses the current positions and velocities to compute new positions and velocities
# 2. The new state becomes the current state for the next iteration
# 3. This process continues for num_steps iterations
# The result is a trajectory showing how the system evolves under the Lennard-Jones potential.
# The Velocity Verlet integrator conserves energy well and is stable for molecular dynamics,
# making it ideal for long simulations. The simulation accumulates the effects of all pairwise
# atomic interactions over many time steps, allowing us to observe emergent dynamics such as
# clustering, oscillations, or equilibration of the system.

def MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps):
    '''This function runs MD simulation for a system of atoms interacting through Lennard Jones potential
    Inputs:
    init_positions: initial positions, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    init_velocities: initial velocities,  2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    m: mass,float
    dt: time step size, float
    num_steps: total steps to run, float
    Outputs:
    curr_positions: final positions of atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    curr_velocities: final velocities of atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    '''
    
    # Helper function to calculate pairwise force
    def f_ij(sigma, epsilon, r):
        r_magnitude = np.sqrt(np.sum(r**2))
        if r_magnitude == 0:
            return np.zeros(3)
        sigma_over_r = sigma / r_magnitude
        f_radial = 24 * epsilon / r_magnitude * (2 * (sigma_over_r**12) - (sigma_over_r**6))
        f = f_radial * (r / r_magnitude)
        return f
    
    # Helper function to aggregate forces
    def aggregate_forces(sigma, epsilon, positions):
        N = positions.shape[0]
        forces = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = positions[i] - positions[j]
                    force_ij = f_ij(sigma, epsilon, r_ij)
                    forces[i] += force_ij
        return forces
    
    # Helper function for Velocity Verlet integration
    def velocity_verlet(sigma, epsilon, positions, velocities, dt, m):
        # Stage 1: Calculate current accelerations
        current_forces = aggregate_forces(sigma, epsilon, positions)
        current_accelerations = current_forces / m
        
        # Stage 2: Update positions using x(t+dt) = x(t) + v(t)*dt + (1/2)*a(t)*dt^2
        new_positions = positions + velocities * dt + 0.5 * current_accelerations * (dt ** 2)
        
        # Stage 3: Calculate accelerations at new positions
        new_forces = aggregate_forces(sigma, epsilon, new_positions)
        new_accelerations = new_forces / m
        
        # Stage 4: Update velocities using v(t+dt) = v(t) + (1/2)*(a(t) + a(t+dt))*dt
        new_velocities = velocities + 0.5 * (current_accelerations + new_accelerations) * dt
        
        return new_positions, new_velocities
    
    # Initialize current state with initial conditions
    curr_positions = np.copy(init_positions)
    curr_velocities = np.copy(init_velocities)
    
    # Run the simulation for num_steps iterations
    for step in range(int(num_steps)):
        # Apply one step of Velocity Verlet integration
        curr_positions, curr_velocities = velocity_verlet(sigma, epsilon, curr_positions, curr_velocities, dt, m)
    
    return curr_positions, curr_velocities


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('51.4', 4)
target = targets[0]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
m = 1
sigma = 1
epsilon = 1
dt = 0.005
N = 32
num_steps = 1000
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
assert np.allclose(MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps), target)
target = targets[1]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
m = 2
sigma = 3
epsilon = 5
dt = 0.01
N = 16
num_steps = 10
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
assert np.allclose(MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps), target)
target = targets[2]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
m = 1
sigma = 1
epsilon = 1
dt = 0.02
N = 8
num_steps = 20
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
assert np.allclose(MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps), target)
target = targets[3]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
def kinetic_energy(velocities, m):
    return 0.5 * m * np.sum(velocities**2)
def potential_energy(positions, epsilon, sigma):
    V = 0.0
    N = len(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            V += 4 * epsilon * ((sigma / r_ij)**12 - (sigma / r_ij)**6)
    return V
def total_energy(positions, velocities, m, epsilon, sigma):
    K = kinetic_energy(velocities, m)
    V = potential_energy(positions, epsilon, sigma)
    return K + V
def linear_momentum(velocities, m):
    return m * np.sum(velocities, axis=0)
def angular_momentum(positions, velocities, m):
    L = np.zeros(3)
    for i in range(len(positions)):
        L += np.cross(positions[i], m * velocities[i])
    return L
m = 1
sigma = 1
epsilon = 1
dt = 0.0025
N = 32
num_steps = 1000
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
positions,velocities = MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps)
# Initial values
initial_energy = total_energy(init_positions, init_velocities, m, epsilon, sigma)
initial_linear_momentum = linear_momentum(init_velocities, m)
initial_angular_momentum = angular_momentum(init_positions, init_velocities, m)
# Final values (after simulation)
final_energy = total_energy(positions, velocities, m, epsilon, sigma)
final_linear_momentum = linear_momentum(velocities, m)
final_angular_momentum = angular_momentum(positions, velocities, m)
# Checking conservation
assert (np.isclose(final_energy,initial_energy), np.allclose(final_linear_momentum, initial_linear_momentum),
        np.allclose(final_angular_momentum, initial_angular_momentum)) == target
