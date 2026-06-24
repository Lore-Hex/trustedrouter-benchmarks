import numpy as np
from math import exp
from scipy.integrate import solve_ivp

# Background: In DNA/RNA ligation chemistry, 2-mers (dimers) consist of two nucleotides.
# Each nucleotide has a complement (A↔A', C↔C', etc.). The complementary 2-mer to "ij" is "j'i'"
# (reverse complement). Given a concentration matrix d organized as [original monomers | complement monomers],
# we need to map each 2-mer position to its complement's position. Since the matrix stores d[i,j] as
# the concentration of 2-mer "ij", the complementary sequence "j'i'" has concentration at position
# dcomp[i,j] = d[j', i'], where i' and j' are the indices of the complementary monomers.
# The complement mapping follows: if index k < Z/2 (original), then k' = k + Z/2 (complement);
# if k >= Z/2 (complement), then k' = k - Z/2 (original).

def GetComp(d):
    '''Concentration matrix of complementary 2-mers
    Inputs:
    d: concentration of 2-mers, numpy float array with dimensions [Z, Z]
    Outputs:
    dcomp: concentration of 2-mers, numpy float array with dimensions [Z, Z], where dcomp[i, j]==d[j', i'].
    '''
    
    Z = d.shape[0]
    half_Z = Z // 2
    
    dcomp = np.zeros_like(d)
    
    for i in range(Z):
        for j in range(Z):
            # Find complement indices
            # If index < Z/2, complement is index + Z/2
            # If index >= Z/2, complement is index - Z/2
            i_comp = i + half_Z if i < half_Z else i - half_Z
            j_comp = j + half_Z if j < half_Z else j - half_Z
            
            # dcomp[i, j] = d[j', i']
            dcomp[i, j] = d[j_comp, i_comp]
    
    return dcomp



# Background: In DNA/RNA ligation chemistry, chains are extended by ligation of 2-mers.
# The master equation describes the time evolution of 2-mer concentrations d_ij:
# d_ij(t) = λ_ij · r_i(t) · l_j(t) · d_j'i'(t) - d_ij(t)
# where:
# - r_i(t) = concentration of all chains ending with monomer i = c_i + sum_k(d_ki)
# - l_j(t) = concentration of all chains starting with monomer j = c_j + sum_k(d_jk)
# - d_j'i'(t) = concentration of complementary 2-mer j'i' (obtained via GetComp)
# - λ_ij = ligation rate for forming 2-mer ij (given as input)
# - The -d_ij(t) term represents spontaneous breakage at rate 1
# During integration, we flatten the 2D matrix to 1D for ODE solver compatibility,
# so we must reshape, compute, and flatten again.

def step(t, y, c, lig):
    '''To integrate the master equation of all the 2-mers
    Inputs:
    t: timepoint, float
    y: flattened current state of the d matrix, numpy float array with length [Z^2]
    c: concentration of monomers of each type, numpy float array with length [Z]
    lig: lambda_{ij} matrix of each 2-mer's ligation rate during the night phase, numpy float array with dimensions [Z, Z]
    Outputs:
    ystep: flattened changing rate of the d matrix, numpy float array with length [Z^2]
    '''
    
    Z = c.shape[0]
    
    # Reshape flattened y back to 2D matrix
    d = y.reshape((Z, Z))
    
    # Compute complementary 2-mer concentration matrix
    dcomp = GetComp(d)
    
    # Compute r_i: concentration of all chains ending with monomer i
    # r_i = c_i + sum_k(d_ki)
    r = c + np.sum(d, axis=0)  # sum over rows for each column
    
    # Compute l_j: concentration of all chains starting with monomer j
    # l_j = c_j + sum_k(d_jk)
    l = c + np.sum(d, axis=1)  # sum over columns for each row
    
    # Compute the master equation: d_ij(t) = λ_ij · r_i · l_j · d_j'i' - d_ij
    # For each element (i, j):
    dddt = np.zeros_like(d)
    for i in range(Z):
        for j in range(Z):
            dddt[i, j] = lig[i, j] * r[i] * l[j] * dcomp[i, j] - d[i, j]
    
    # Flatten back to 1D for ODE solver
    ystep = dddt.flatten()
    
    return ystep


def GetComp(d):
    '''Concentration matrix of complementary 2-mers
    Inputs:
    d: concentration of 2-mers, numpy float array with dimensions [Z, Z]
    Outputs:
    dcomp: concentration of 2-mers, numpy float array with dimensions [Z, Z], where dcomp[i, j]==d[j', i'].
    '''
    
    Z = d.shape[0]
    half_Z = Z // 2
    
    dcomp = np.zeros_like(d)
    
    for i in range(Z):
        for j in range(Z):
            # Find complement indices
            # If index < Z/2, complement is index + Z/2
            # If index >= Z/2, complement is index - Z/2
            i_comp = i + half_Z if i < half_Z else i - half_Z
            j_comp = j + half_Z if j < half_Z else j - half_Z
            
            # dcomp[i, j] = d[j', i']
            dcomp[i, j] = d[j_comp, i_comp]
    
    return dcomp


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('44.2', 3)
target = targets[0]

t = 0
y = np.array([[1, 2], [3, 4]]).flatten()
c = np.ones(2)*10
lig = np.array([[1, 0], [0, 1]])
assert np.allclose(step(t, y, c, lig), target)
target = targets[1]

t = 0
y = np.array([[0.9204365 , 0.99516558, 0.76198302, 0.50127276],
       [0.78900999, 0.53480707, 0.68065993, 0.20993823],
       [0.32379346, 0.29234994, 1.20530971, 0.6083708 ],
       [1.33007112, 0.37400537, 5.03614455, 0.79329857]]).flatten()
c = np.ones(4)*10
lig = np.array([[0.8997858 , 1.01790812, 1.23096801, 0.59830888],
       [1.34481097, 1.33982066, 0.72675653, 0.78848685],
       [0.79413525, 0.97270199, 0.96545486, 0.80494749],
       [0.87638968, 1.09389482, 0.75718173, 1.09107295]])
assert np.allclose(step(t, y, c, lig), target)
target = targets[2]

t = 20
y = np.array([[0.9204365 , 0.99516558, 0.76198302, 0.50127276],
       [0.78900999, 0.53480707, 0.68065993, 0.20993823],
       [0.32379346, 0.29234994, 1.20530971, 0.6083708 ],
       [1.33007112, 0.37400537, 5.03614455, 0.79329857]]).flatten()
c = np.ones(4)*10
lig = np.eye(4)
assert np.allclose(step(t, y, c, lig), target)
