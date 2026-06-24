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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('38.1', 3)
target = targets[0]

a,b = np.array([3,4,5]),np.array([4,3,2])
assert np.allclose(cross(a,b), target)
target = targets[1]

a,b = np.array([3,4,7]),np.array([8,2,6])
assert np.allclose(cross(a,b), target)
target = targets[2]

a,b = np.array([1,1,0]),np.array([1,-1,0])
assert np.allclose(cross(a,b), target)
