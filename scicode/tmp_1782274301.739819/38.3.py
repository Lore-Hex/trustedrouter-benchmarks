import numpy as np

# Background: The cross product of two 3D vectors is a binary operation that produces a vector perpendicular to both input vectors.
# For vectors a = [a1, a2, a3] and b = [b1, b2, b3], the cross product is calculated as:
# a × b = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
# The magnitude of the cross product equals the area of the parallelogram spanned by the two vectors.
# The direction follows the right-hand rule: if you curl your right hand's fingers from vector a to vector b,
# your thumb points in the direction of a × b.
# The cross product is anti-commutative: a × b = -(b × a)
# NumPy provides a built-in function np.cross() that computes this efficiently.

def cross(a, b):
    '''Calculates the cross product of the input vectors.
    Input:
    a (numpy array): Vector a.
    b (numpy array): Vector b.
    Output:
    t (numpy array): The cross product of a and b.
    '''
    
    t = np.cross(a, b)
    return t


# Background: Reciprocal vectors (also called dual vectors or reciprocal lattice vectors) are used in crystallography, physics, and signal processing.
# Given three linearly independent vectors a1, a2, a3 in 3D space, their reciprocal vectors b1, b2, b3 satisfy:
# a_i · b_j = δ_ij (Kronecker delta: 1 if i==j, 0 if i!=j)
# The reciprocal vectors are calculated as:
# b1 = (a2 × a3) / (a1 · (a2 × a3))
# b2 = (a3 × a1) / (a2 · (a3 × a1))
# b3 = (a1 × a2) / (a3 · (a1 × a2))
# The denominator (a1 · (a2 × a3)) is the scalar triple product and represents the volume of the parallelepiped formed by the three vectors.
# Reciprocal vectors are useful in Fourier analysis, crystallographic calculations, and defining dual bases in linear algebra.

def reciprocal_3D(a1, a2, a3):
    '''Calculates the 3D reciprocal vectors given the input.
    Input:
    a1 (numpy array): Vector 1.
    a2 (numpy array): Vector 2.
    a3 (numpy array): Vector 3.
    Returns:
    b_i (list of numpy arrays): The collection of reciprocal vectors.
    '''
    
    # Calculate the scalar triple product (volume of parallelepiped)
    cross_a2_a3 = np.cross(a2, a3)
    volume = np.dot(a1, cross_a2_a3)
    
    # Calculate reciprocal vectors using the formula
    # b1 = (a2 × a3) / (a1 · (a2 × a3))
    b1 = cross_a2_a3 / volume
    
    # b2 = (a3 × a1) / (a2 · (a3 × a1))
    cross_a3_a1 = np.cross(a3, a1)
    b2 = cross_a3_a1 / volume
    
    # b3 = (a1 × a2) / (a3 · (a1 × a2))
    cross_a1_a2 = np.cross(a1, a2)
    b3 = cross_a1_a2 / volume
    
    b_i = [b1, b2, b3]
    return b_i



# Background: Reciprocal vectors (also called dual vectors) are used in crystallography, physics, and signal processing.
# For a single vector a in 3D space, the reciprocal vector is typically defined as a* = a / (a · a), which is the unit vector scaled inversely by the squared magnitude.
# For two linearly independent vectors a1 and a2 in 3D space, their reciprocal vectors are defined such that a_i · b_j = δ_ij (Kronecker delta).
# The reciprocal vectors for two 3D vectors a1, a2 can be computed by:
#   - Finding a third vector a3 perpendicular to the plane spanned by a1 and a2: a3 = a1 × a2
#   - Then applying the 3D reciprocal formula to get b1, b2, b3
#   - Returning only the reciprocals corresponding to the input vectors (b1, b2)
# For three linearly independent vectors a1, a2, a3, the reciprocal vectors are:
#   - b1 = (a2 × a3) / (a1 · (a2 × a3))
#   - b2 = (a3 × a1) / (a2 · (a3 × a1))
#   - b3 = (a1 × a2) / (a3 · (a1 × a2))
# where the denominator is the scalar triple product (volume of the parallelepiped).

def reciprocal(*args):
    '''Computes the reciprocal vector(s) based on the input vector(s).
    Input:
    *args (numpy array(s)): some vectors, with the amount uncertain within the range of [1, 2, 3]
    Output:
    rec (list of numpy array(s)): The collection of all the reciprocal vectors.
    '''
    
    num_vectors = len(args)
    
    if num_vectors == 1:
        # For a single vector, reciprocal is a / (a · a)
        a = args[0]
        magnitude_sq = np.dot(a, a)
        rec = [a / magnitude_sq]
    
    elif num_vectors == 2:
        # For two vectors, we need to construct a third perpendicular vector
        a1 = args[0]
        a2 = args[1]
        # a3 is perpendicular to both a1 and a2
        a3 = np.cross(a1, a2)
        
        # Calculate the scalar triple product (volume)
        volume = np.dot(a1, np.cross(a2, a3))
        
        # Calculate reciprocal vectors
        # b1 = (a2 × a3) / volume
        b1 = np.cross(a2, a3) / volume
        # b2 = (a3 × a1) / volume
        b2 = np.cross(a3, a1) / volume
        
        # Return only the reciprocals for the input vectors
        rec = [b1, b2]
    
    elif num_vectors == 3:
        # For three vectors, use the standard 3D reciprocal formula
        a1 = args[0]
        a2 = args[1]
        a3 = args[2]
        
        # Calculate the scalar triple product (volume of parallelepiped)
        cross_a2_a3 = np.cross(a2, a3)
        volume = np.dot(a1, cross_a2_a3)
        
        # Calculate reciprocal vectors
        b1 = cross_a2_a3 / volume
        cross_a3_a1 = np.cross(a3, a1)
        b2 = cross_a3_a1 / volume
        cross_a1_a2 = np.cross(a1, a2)
        b3 = cross_a1_a2 / volume
        
        rec = [b1, b2, b3]
    
    return rec


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('38.3', 4)
target = targets[0]

a1,a2,a3 = np.array([1,1,0]),np.array([1,-1,0]),np.array([1,0,2])
rec = reciprocal(a1,a2,a3)
assert (np.isclose(2*np.pi,a1 @ rec[0])) == target
target = targets[1]

a1,a2,a3 = np.array([1,1,0]),np.array([1,-1,2]),np.array([1,3,5])
assert np.allclose(reciprocal(a1,a2,a3), target)
target = targets[2]

a1,a2 = np.array([1,4,0]),np.array([2,-1,0])
assert np.allclose(reciprocal(a1,a2), target)
target = targets[3]

a1 = np.array([1,1,5])
assert np.allclose(reciprocal(a1), target)
