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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('51.1', 3)
target = targets[0]

sigma = 1
epsilon = 1
r = np.array([-3.22883506e-03,  2.57056485e+00,  1.40822287e-04])
assert np.allclose(f_ij(sigma,epsilon,r), target)
target = targets[1]

sigma = 2
epsilon = 3
r = np.array([3,  -4,  5])
assert np.allclose(f_ij(sigma,epsilon,r), target)
target = targets[2]

sigma = 3
epsilon = 7
r = np.array([5,  9,  7])
assert np.allclose(f_ij(sigma,epsilon,r), target)
