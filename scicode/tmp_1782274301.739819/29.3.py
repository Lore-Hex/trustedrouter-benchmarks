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



# Background: Gram-Schmidt orthogonalization is a process that transforms a set of linearly independent vectors
# into a set of orthogonal (and optionally orthonormal) vectors. The algorithm works iteratively:
# 1. Start with the first vector and normalize it to unit length.
# 2. For each subsequent vector, subtract its projections onto all previously computed orthonormal vectors.
#    This removes any component in the direction of those vectors.
# 3. Normalize the resulting vector to unit length.
# The projection of vector v onto unit vector u is: proj_u(v) = <v,u> * u
# where <v,u> is the inner product (dot product) of v and u.
# The Gram-Schmidt process ensures that the output vectors form an orthonormal basis:
# - Orthogonal: <u_i, u_j> = 0 for i ≠ j
# - Normalized: ||u_i|| = 1 for all i
# This is fundamental in linear algebra, numerical methods, and machine learning (e.g., QR decomposition,
# Principal Component Analysis). The process is numerically sensitive, so care must be taken with floating-point
# arithmetic, particularly when vectors are nearly linearly dependent.

def orthogonalize(A):
    '''Perform Gram-Schmidt orthogonalization on the input vectors to produce orthogonal and normalized vectors.
    Input:
    A (N*N numpy array): N linearly independent vectors in the N-dimension space.
    Output:
    B (N*N numpy array): The collection of the orthonormal vectors.
    '''
    
    # Get the number of vectors (N)
    N = A.shape[1]
    
    # Initialize output array with the same shape as input
    B = np.zeros_like(A, dtype=float)
    
    # Process each vector
    for i in range(N):
        # Start with the i-th vector from the input
        v = A[:, i].astype(float)
        
        # Subtract projections onto all previously computed orthonormal vectors
        for j in range(i):
            # Compute projection of v onto the j-th orthonormal vector
            # proj = <v, B[:, j]> * B[:, j]
            projection = np.dot(v, B[:, j]) * B[:, j]
            v = v - projection
        
        # Normalize the resulting vector to unit length
        # Handle the case where the vector becomes zero due to linear dependence
        norm = np.linalg.norm(v)
        if norm > 1e-10:  # Use small threshold to handle numerical errors
            B[:, i] = v / norm
        else:
            B[:, i] = v
    
    return B


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('29.3', 4)
target = targets[0]

A = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]]).T
B = orthogonalize(A)
assert (np.isclose(0,B[:,0].T @ B[:,3])) == target
target = targets[1]

A = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]]).T
assert np.allclose(orthogonalize(A), target)
target = targets[2]

A = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]]).T
assert np.allclose(orthogonalize(A), target)
target = targets[3]

A = np.array([[0, 1, 1, -1, 1], [1, 0, -1, 1, -1], [1, -1, 0, -1, -1], [-1, 1, -1, 0, -1], [-1, -1, -1, -1, 0]]).T
assert np.allclose(orthogonalize(A), target)
