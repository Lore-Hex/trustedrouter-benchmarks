import numpy as np

# Background: Newton's law of universal gravitation states that the gravitational force between two particles
# is proportional to the product of their masses and inversely proportional to the square of the distance between them.
# The force vector points from particle i toward particle j (or vice versa depending on perspective).
# For each particle i, we must sum the gravitational forces exerted by all other particles j.
# The force on particle i due to particle j is: F_ij = -G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
# This is a vector equation where (r_j - r_i) is the displacement vector from i to j.
# The negative sign indicates the force is attractive (points from i toward j when considering the direction of acceleration).
# G = 6.67430e-11 N m^2/kg^2 is the gravitational constant.

def Nbody_Fij(xin, mass):
    '''This function computes the forces between particles subject to gravity.
    Inputs:
    xin: 2D array of locations for the N particles, N,3 elements where 3 is x,y,z coordinates
    mass: 1D array of particle masses, N elements
    Outputs:
    f: 2D array of forces, on each particle, N,3 elements where 3 is x,y,z coordinates
    '''
    
    G = 6.67430e-11  # Gravitational constant in N m^2/kg^2
    
    N = xin.shape[0]  # Number of particles
    f = np.zeros((N, 3))  # Initialize force array
    
    # For each particle i, compute the net gravitational force from all other particles j
    for i in range(N):
        for j in range(N):
            if i != j:  # Avoid self-interaction
                # Displacement vector from particle i to particle j
                r_ij = xin[j] - xin[i]  # shape (3,)
                
                # Distance between particles i and j
                r_dist = np.linalg.norm(r_ij)  # scalar
                
                # Gravitational force on particle i due to particle j
                # F_ij = -G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
                # Simplify: F_ij = -G * m_i * m_j * r_ij / r_dist^3
                f[i] += -G * mass[i] * mass[j] * r_ij / (r_dist ** 3)
    
    return f



# Background: The evolution of an N-body gravitational system is governed by Newton's second law (F = ma).
# For a system with N particles, we need to track both positions and velocities over time.
# The right-hand side (RHS) of the evolution equations describes how the state changes:
# - The rate of change of position is simply the velocity: dr/dt = v
# - The rate of change of velocity is the acceleration: dv/dt = a = F/m
# where F is the net gravitational force computed from Nbody_Fij and a = F/m is the acceleration.
# The state vector uin contains both position and velocity information for each particle,
# and the RHS output should return the time derivatives of these quantities in the same format.
# This function serves as the core component for time-stepping integrators (e.g., Runge-Kutta methods).

def Nbody_RHS(uin, mass):
    '''This function computes the right hand side for the evolution equation of a N particle system evolving under gravity. It takes as input the current state uin, and mass mass.
    Inputs:
    uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the x,y,z position of the particle followed by the vx,vy,vz velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    Outputs:
    rhs: a 2D array of velocities and accelerations, N,6 elements; each element is a float; laid out as vx, vy, vz, ax, ay, az.
    '''
    
    N = uin.shape[0]  # Number of particles
    rhs = np.zeros((N, 6))  # Initialize RHS array
    
    # Extract positions from uin (first 3 columns)
    positions = uin[:, 0:3]  # N x 3 array
    
    # Extract velocities from uin (second 3 columns)
    velocities = uin[:, 3:6]  # N x 3 array
    
    # Compute gravitational forces using Nbody_Fij
    forces = Nbody_Fij(positions, mass)  # N x 3 array
    
    # Compute accelerations: a = F/m
    accelerations = forces / mass[:, np.newaxis]  # N x 3 array
    
    # RHS for positions is velocity
    rhs[:, 0:3] = velocities
    
    # RHS for velocities is acceleration
    rhs[:, 3:6] = accelerations
    
    return rhs


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('49.2', 3)
target = targets[0]

uin = np.array([[0.28382835, 0.60608318, 0.94422514, 0.85273554, 0.00225923,
        0.52122603],
       [0.55203763, 0.48537741, 0.76813415, 0.16071675, 0.76456045,
        0.0208098 ]])
mass = np.array([0.17916907, 0.91202765])
assert np.allclose(Nbody_RHS(uin, mass), target)
target = targets[1]

uin = np.array([[0.13521018, 0.11627302, 0.30989758, 0.67145265, 0.47122978,
        0.8161683 ],
       [0.28958678, 0.73312598, 0.70262236, 0.32756948, 0.33464753,
        0.97805808],
       [0.62458211, 0.95031352, 0.76747565, 0.82500925, 0.4066403 ,
        0.45130841]])
mass = np.array([0.26331502, 0.53373939, 0.01457496])
assert np.allclose(Nbody_RHS(uin, mass), target)
target = targets[2]

uin = np.array([[0.40063163, 0.99513816, 0.17756418, 0.9625969 , 0.41925027,
        0.42405245],
       [0.46314887, 0.37372315, 0.4655081 , 0.03516826, 0.08427267,
        0.7325207 ],
       [0.63619999, 0.02790779, 0.30017006, 0.22085252, 0.05501999,
        0.52324607],
       [0.41636966, 0.04821875, 0.57240452, 0.80351665, 0.11360172,
        0.27711721],
       [0.64167809, 0.49273731, 0.50655608, 0.46144026, 0.89474307,
        0.60577769]])
mass = np.array([0.91874701, 0.90071485, 0.03342143, 0.95694934, 0.13720932])
assert np.allclose(Nbody_RHS(uin, mass), target)
