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


# Background: The 4th order classical Runge-Kutta (RK4) method is a powerful time-stepping technique
# for solving ordinary differential equations (ODEs) of the form du/dt = f(u, t).
# It achieves 4th order accuracy by evaluating the right-hand side at four intermediate points
# within each time step and combining them with specific weights.
# For a time step from t to t + dt, the RK4 method computes:
# k1 = f(u_n, t_n)                               (slope at the start)
# k2 = f(u_n + 0.5*dt*k1, t_n + 0.5*dt)         (slope at the midpoint using k1)
# k3 = f(u_n + 0.5*dt*k2, t_n + 0.5*dt)         (slope at the midpoint using k2)
# k4 = f(u_n + dt*k3, t_n + dt)                 (slope at the endpoint using k3)
# The updated state is then: u_{n+1} = u_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
# This weighted average of the slopes provides excellent accuracy for smooth functions.
# In the context of N-body gravitational systems, the RHS is computed by Nbody_RHS,
# which returns both velocity (as RHS of position) and acceleration (as RHS of velocity).

def Nbody_RK4(uin, mass, dt):
    '''This function takes a single step of size dt computing an updated state vector uout 
    given the current state uin, mass, and timestep dt.
    Inputs:
    uin: a 2D array of current positions and velocities. N,6 elements; each element is a float; 
         each particle is described by its location and velocity, first the x,y,z position of 
         the particle followed by the vx,vy,vz velocity of the particle. Thus particle "i"'s 
         position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    dt: time step to use for the evolution; a float
    Outputs:
    uout: a 2D array of final positions and velocities, N,6 elements; each element is a float; 
          laid out like uin
    '''
    
    # Compute the four RK4 slopes
    k1 = Nbody_RHS(uin, mass)
    
    k2 = Nbody_RHS(uin + 0.5 * dt * k1, mass)
    
    k3 = Nbody_RHS(uin + 0.5 * dt * k2, mass)
    
    k4 = Nbody_RHS(uin + dt * k3, mass)
    
    # Combine the slopes using RK4 weights: (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    uout = uin + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return uout



# Background: The n-body simulation integrates the equations of motion for N particles
# subject to gravitational forces. The simulation starts at time t0 and evolves the system
# to time t1 using discrete time steps of size dt. At each time step, the classical 4th-order
# Runge-Kutta method (RK4) is applied to update both positions and velocities. The state
# vector tracks the position (x, y, z) and velocity (vx, vy, vz) for each particle. The
# simulation proceeds by repeatedly calling the RK4 integrator, advancing the current time
# by dt with each step, until the final time t1 is reached. The output is the final state
# of the system at t1 (or very close to it, depending on whether (t1 - t0) is an exact
# multiple of dt).

def Nbody(uin, mass, dt, t0, t1):
    '''This function runs n-body simulation for N particles interacting via gravity only.
    Inputs:
    uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the x,y,z position of the particle followed by the vx,vy,vz velocity of the particle. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    dt: time step size; a float
    t0: initial time; a float
    t1: final time; a float
    Outputs:
    uout: final positions and velocities of particles; an array of floats laid out like uin
    '''
    
    # Initialize current state with the initial conditions
    u = uin.copy()
    
    # Initialize current time
    t = t0
    
    # Time-stepping loop: integrate from t0 to t1 using fixed time steps of size dt
    while t < t1:
        # Determine the next time step size
        # If the remaining time is less than dt, use the remaining time as the step size
        dt_step = min(dt, t1 - t)
        
        # Perform one RK4 step to advance the state
        u = Nbody_RK4(u, mass, dt_step)
        
        # Update current time
        t += dt_step
    
    # Return the final state at time t1
    uout = u
    
    return uout


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('49.4', 4)
target = targets[0]

uin = np.array([[0.6033556 , 0.44387186, 0.48004678, 0.88844753, 0.20850049,
        0.94458146],
       [0.07347004, 0.59515246, 0.03115107, 0.66525743, 0.6373855 ,
        0.86246516]])
mass = np.array([0.08463852, 0.31566824])
dt = 0.5
t0 = 0
t1 = 60.
assert np.allclose(Nbody(uin, mass, dt, t0, t1), target)
target = targets[1]

uin = np.array([[0.94163771, 0.44562705, 0.66995763, 0.9243307 , 0.61942347,
        0.32258573],
       [0.31947631, 0.29145669, 0.95741742, 0.40595834, 0.94655582,
        0.85719056],
       [0.68892767, 0.00328797, 0.9001035 , 0.91986487, 0.00590936,
        0.64292615]])
mass = np.array([0.00800334, 0.0104384 , 0.5974588 ])
dt = 0.3
t0 = 30
t1 = 60
assert np.allclose(Nbody(uin, mass, dt, t0, t1), target)
target = targets[2]

uin = np.array([[0.38521698, 0.59574189, 0.61080706, 0.59961919, 0.312284  ,
        0.06987367],
       [0.80073423, 0.91143405, 0.19467463, 0.21060764, 0.37744078,
        0.40284721],
       [0.88517769, 0.20215434, 0.41901483, 0.64795917, 0.51164295,
        0.33383249],
       [0.77847343, 0.13543513, 0.19171077, 0.40258915, 0.78684447,
        0.25380819],
       [0.10948542, 0.93067601, 0.12638728, 0.60500539, 0.29012194,
        0.62652939]])
mass = np.array([0.65947595, 0.06772616, 0.35544292, 0.33472408, 0.17750637])
dt = 0.15
t0 = 0.
t1 = 60.*10.
assert np.allclose(Nbody(uin, mass, dt, t0, t1), target)
target = targets[3]

'''
This function computes the conserved total eenrgy for N particles interacting via gravity only.
Inputs:
uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
mass: an 1D array of particle masses; an array of floats
Outputs:
Etot: the conserved total energy, float
'''
def Nbody_Etot(uin, mass):
    Ekin = 0.5*np.sum(mass * (uin[:,3]**2 + uin[:,4]**2 + uin[:,5]**2))
    # this is horrible, but somewhat close to the naive expression given above
    ggrav = 6.67430e-11
    Epot = 0.
    for i in range(len(mass)):
        for j in range(i+1, len(mass)):
            rij = np.sqrt((uin[i,0] - uin[j,0])**2 + (uin[i,1] - uin[j,1])**2 + (uin[i,2] - uin[j,2])**2)
            Epot += -2. * ggrav * mass[i] * mass[j] / rij
    Etot = Ekin + Epot
    return Etot
# mass of Sun and Earth
mass = np.array([1.98855e30, 5.97219e24])
# initial position and velocity (Sun at origin and at rest, Earth velocity along y axis)
uin = np.array([[0.,0.,0.,
                 0.,0.,0.],
                [1.49598261e11,0.,0.,
                 0.,29.78e3,0.]])
t0 = 0.
t1 = 365.*24.*60.*60. # 1 year
dt = 60.*60.          # 1 hour
Nbody(uin, mass, dt, t0, t1)
Etot0 = Nbody_Etot(uin, mass)
uout = Nbody(uin, mass, dt, t0, t1)
Etot1 = Nbody_Etot(uout, mass)
assert (np.isclose(Etot0, Etot1)) == target
