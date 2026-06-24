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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('51.3', 3)
target = targets[0]

sigma = 1
epsilon = 1
positions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])
velocities = np.array([[0.78759462, 0.55733813, 0.97321828],
       [0.67351598, 0.74988225, 0.2430374 ]])
dt = 0.005
m = 1
assert np.allclose(velocity_verlet(sigma,epsilon,positions,velocities,dt,m), target)
target = targets[1]

sigma = 2
epsilon = 3
positions = np.array([[2.15569639, 3.33938018, 0.1682584 ],[0.20055338, 5.56353376, 1.9202491], [0.61941565, 0.19380897, 0.20261979 ]])
velocities = np.array([[0.90594996, 0.96932907, 0.57531663],
       [0.80310603, 0.96969681, 0.1771265 ],
       [0.87580384, 0.42642859, 0.40785355]])
dt = 0.01
m = 3
assert np.allclose(velocity_verlet(sigma,epsilon,positions,velocities,dt,m), target)
target = targets[2]

sigma = 5
epsilon = 2
positions = np.array([[5.62726631, 5.3077771 , 7.29719649],
       [7.25031287, 7.58926428, 3.71262908],
       [8.7866416 , 3.73724676, 9.22676027],
       [7.89096788, 5.3872004 , 7.95350911],
       [6.068183  , 3.55807037, 2.7965242 ]])
velocities = np.array([[0.21901831, 0.03442682, 0.56648053],
       [0.70408383, 0.89261961, 0.08422311],
       [0.43673835, 0.28973691, 0.74121155],
       [0.06514819, 0.59291869, 0.7190472 ],
       [0.40667563, 0.28100076, 0.08050432]])
dt = 0.2
m = 5
assert np.allclose(velocity_verlet(sigma,epsilon,positions,velocities,dt,m), target)
