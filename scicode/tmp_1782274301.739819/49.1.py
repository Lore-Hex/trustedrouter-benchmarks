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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('49.1', 3)
target = targets[0]

xin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
mass = np.array([1.0e4, 1.0e4])
assert np.allclose(Nbody_Fij(xin, mass), target)
target = targets[1]

xin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(0.75), 0.0]])
mass = np.array([1.0e3, 1.5e4, 2.0e5])
assert np.allclose(Nbody_Fij(xin, mass), target)
target = targets[2]

xin = np.array([[0.89660964, 0.20404593, 0.26538925],
       [0.83776986, 0.60298593, 0.26242915],
       [0.47325228, 0.77652396, 0.4536616 ],
       [0.11320759, 0.9165638 , 0.8266307 ],
       [0.4766697 , 0.56786154, 0.0529977 ]])
mass = np.array([0.72386384e7, 0.73792416e3, 0.78105276e2, 0.30785728e5, 0.59321658e6])
assert np.allclose(Nbody_Fij(xin, mass), target)
