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



# Background: The inner product (also called dot product) is a fundamental operation in linear algebra
# that measures the similarity or projection between two vectors in N-dimensional space.
# For vectors u and v in R^n, the inner product is defined as: <u,v> = sum(u_i * v_i) for all components.
# The inner product has several important properties:
# 1. It produces a scalar value (not a vector).
# 2. If two vectors are orthogonal (perpendicular), their inner product is zero.
# 3. The inner product of a vector with itself gives the squared L2 norm: <v,v> = ||v||^2.
# 4. Inner product is commutative: <u,v> = <v,u>.
# 5. It is linear in both arguments: <au + bw, v> = a<u,v> + b<w,v>.
# The inner product is essential in machine learning for computing similarity measures,
# projections, and as the basis for algorithms like cosine similarity and Support Vector Machines.

def inner_product(u, v):
    '''Calculates the inner product of two vectors.
    Input:
    u (numpy array): Vector 1.
    v (numpy array): Vector 2.
    Output:
    p (float): Inner product of the vectors.
    '''
    
    # Compute the inner product using numpy's dot product function
    # This efficiently computes the sum of element-wise products
    p = np.dot(u, v)
    
    return p


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('29.2', 3)
target = targets[0]

u,v = np.array([3,4]),np.array([4,3])
assert np.allclose(inner_product(u,v), target)
target = targets[1]

u,v = np.array([3,4]),np.array([3,4])
assert np.allclose(inner_product(u,v), target)
target = targets[2]

u,v = np.array([3,4,7,6]),np.array([4,3,2,8])
assert np.allclose(inner_product(u,v), target)
