import numpy as np



# Background: Vector normalization (also called unit vector creation) is a fundamental operation in linear algebra and machine learning.
# Normalization scales a vector to have a magnitude (Euclidean norm or L2 norm) of 1, while preserving its direction.
# The L2 norm of a vector v is calculated as: ||v|| = sqrt(sum(v_i^2)) for all components v_i.
# To normalize, divide each component by the L2 norm: n = v / ||v||.
# This is essential in machine learning for feature scaling, similarity computations, and ensuring numerical stability.

def normalize(v):
    '''Normalize the input vector.
    Input:
    v (N*1 numpy array): The input vector.
    Output:
    n (N*1 numpy array): The normalized vector.
    '''
    
    # Calculate the L2 norm (Euclidean norm) of the vector
    norm = np.linalg.norm(v)
    
    # Avoid division by zero: if norm is zero, return the zero vector
    if norm == 0:
        return v
    
    # Divide each component by the norm to get the unit vector
    n = v / norm
    
    return n


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('29.1', 3)
target = targets[0]

v = np.array([3,4])
assert np.allclose(normalize(v), target)
target = targets[1]

v = np.array([1,2,3,4,5,6,7,8]).reshape(4,2)
assert np.allclose(normalize(v), target)
target = targets[2]

v = np.array([i for i in range(12)]).reshape(3,2,2)
assert np.allclose(normalize(v), target)
