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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('38.2', 3)
target = targets[0]

a1,a2,a3 = np.array([3,4,5]),np.array([4,3,2]),np.array([1,0,2])
assert np.allclose(reciprocal_3D(a1,a2,a3), target)
target = targets[1]

a1,a2,a3 = np.array([3,4,7]),np.array([8,2,6]),np.array([1,0,2])
assert np.allclose(reciprocal_3D(a1,a2,a3), target)
target = targets[2]

a1,a2,a3 = np.array([1,1,0]),np.array([1,-1,0]),np.array([1,0,2])
assert np.allclose(reciprocal_3D(a1,a2,a3), target)
