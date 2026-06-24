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



# Background: The Boltzmann-Shannon entropy quantifies disorder in the system based on the 
# probability distribution of 2-mer concentrations. Given relative concentrations 
# tilde_d_ij = d_ij / sum_kl(d_kl), the entropy is calculated as:
# S = -sum_ij(tilde_d_ij * ln(tilde_d_ij)), where we treat ln(0) = 0 (since lim x→0 x*ln(x) = 0).
# To simulate the dynamics, we integrate the master equation using the previously defined step() function
# over nsteps equally spaced timepoints from 0 to tf. At each timepoint, we compute the relative
# concentrations from the absolute concentrations d_ij, then calculate the Boltzmann-Shannon entropy.
# The process shows entropy decrease as the system self-organizes through ligation and breakage.

def entropies(c0, d0, lig, tf, nsteps):
    '''To calculate the entropy of the system at each timestep
    Inputs:
    c0: concentration of monomers of each type at time 0, numpy float array with length [Z]
    d0: concentration of 2-mers of each type at time 0, numpy float array with dimensions [Z, Z]
    lig: lambda_{ij} matrix of each 2-mer's ligation rate during the night phase, numpy float array with dimensions [Z, Z]
    tf: the simulation goes from 0 to tf, float
    nsteps: the number of simulation steps, int
    Outputs:
    entropy_list: entropy at each timestep, numpy array of length nsteps
    '''
    
    Z = c0.shape[0]
    
    # Flatten the initial 2-mer concentration matrix for ODE solver
    y0 = d0.flatten()
    
    # Create time points for evaluation
    t_eval = np.linspace(0, tf, nsteps)
    
    # Solve the ODE using scipy's solve_ivp
    # The step function is defined from the previous step
    solution = solve_ivp(step, [0, tf], y0, args=(c0, lig), t_eval=t_eval, method='RK45')
    
    # Initialize entropy list
    entropy_list = np.zeros(nsteps)
    
    # Calculate entropy at each timepoint
    for idx, t in enumerate(t_eval):
        # Get the d matrix at this timepoint
        d_current = solution.y[:, idx].reshape((Z, Z))
        
        # Calculate total concentration of all 2-mers
        total_concentration = np.sum(d_current)
        
        # Avoid division by zero
        if total_concentration <= 0:
            entropy_list[idx] = 0.0
        else:
            # Calculate relative concentrations (probabilities)
            tilde_d = d_current / total_concentration
            
            # Calculate Boltzmann-Shannon entropy: S = -sum(tilde_d * ln(tilde_d))
            # Treat ln(0) = 0 by using a mask for positive values
            entropy = 0.0
            for i in range(Z):
                for j in range(Z):
                    if tilde_d[i, j] > 0:
                        entropy -= tilde_d[i, j] * np.log(tilde_d[i, j])
            
            entropy_list[idx] = entropy
    
    return entropy_list


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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('44.3', 3)
target = targets[0]

c0 = np.ones(4)*10.0
d0 = np.ones([4, 4])*0.5
lig = np.hstack([np.ones([4, 2]), -np.ones([4, 2])])
tf = 20
nsteps = 1000
assert np.allclose(entropies(c0, d0, lig, tf, nsteps), target)
target = targets[1]

c0 = np.ones(8)
d0 = np.ones([8, 8])*0.01
lig = np.array([[1.21663772, 0.89248457, 1.21745463, 1.11149841, 0.81367134,
        0.89963613, 1.30729711, 1.12344434],
       [0.7829948 , 0.97670401, 1.17241282, 1.17666507, 0.99661672,
        0.82597011, 1.26869751, 0.92559714],
       [1.32370699, 0.81555449, 1.24539873, 1.11920104, 0.70893291,
        0.94277028, 0.99222404, 1.28004994],
       [1.33605707, 0.9677957 , 0.87545897, 1.11743411, 1.00387065,
        1.21387094, 1.3587547 , 1.07888797],
       [1.09560243, 1.22427195, 0.91200443, 0.79503969, 1.32678448,
        1.0089972 , 1.40026918, 1.1007811 ],
       [1.07280747, 1.10334304, 0.69314364, 1.47071329, 1.20877084,
        0.70031823, 0.90295041, 0.8843919 ],
       [0.89329015, 0.97337332, 1.16629781, 1.00359901, 1.05891634,
        0.68708136, 1.26150524, 0.87881251],
       [1.47199815, 0.69658061, 0.84842985, 1.22751336, 1.01383103,
        0.99759963, 0.69677971, 0.99616318]])
tf = 1000
nsteps = 10000
assert np.allclose(entropies(c0, d0, lig, tf, nsteps), target)
target = targets[2]

c0 = np.array([7.0681359 , 2.62361004, 1.59922841, 4.01258308, 0.89411538,
       2.94376844, 1.85685374, 3.45821068])
d0 = np.ones([8, 8])*0.01
lig = np.array([[0.77124931, 0.95367699, 0.98325401, 1.08071122, 1.19929357,
        1.91809708, 0.71432156, 1.27096494],
       [0.85762676, 1.0169952 , 1.18012769, 1.33453853, 1.391186  ,
        1.08892087, 0.96188594, 1.01095109],
       [1.0255774 , 0.97554961, 0.94428129, 0.64227944, 0.73790089,
        0.81056697, 0.91220593, 1.54502225],
       [0.8066888 , 0.81817817, 0.91694827, 1.00399813, 0.82554822,
        0.92915596, 0.83144055, 1.40705141],
       [0.7386233 , 0.82211199, 1.2920814 , 1.1757341 , 1.06815192,
        0.84285394, 1.71679303, 0.60512826],
       [1.16674491, 1.07334837, 1.0456343 , 0.85723942, 1.33040205,
        0.80796219, 0.73944967, 1.21658198],
       [1.03109573, 1.07236791, 0.94173481, 1.17093906, 0.77731087,
        1.02610659, 0.78401785, 0.82595097],
       [1.40816645, 1.12174143, 0.89368929, 1.10921353, 1.06705567,
        0.78623361, 0.97364678, 1.05773036]])
tf = 1000
nsteps = 10000
assert np.allclose(entropies(c0, d0, lig, tf, nsteps), target)
