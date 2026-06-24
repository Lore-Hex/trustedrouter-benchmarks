import numpy as np
import time

# Background: The forced, damped single pendulum is governed by a second-order nonlinear
# differential equation. The equation of motion combines three main forces:
# 1. Restoring force: -g/L * sin(theta) (gravitational component)
# 2. Damping force: -beta * omega (proportional to angular velocity)
# 3. Driving force: A * cos(alpha * t) (external periodic forcing)
# 
# The system is described by:
# d²θ/dt² = -(g/L)*sin(θ) - β*(dθ/dt) + (A/L)*cos(α*t)
#
# To solve this second-order ODE numerically, we convert it to a system of
# two first-order ODEs by defining the state vector y = [θ, ω]^T where:
# θ is the angular displacement (radians)
# ω is the angular velocity (radians/second)
#
# The system becomes:
# dθ/dt = ω
# dω/dt = -(g/L)*sin(θ) - β*ω + (A/L)*cos(α*t)
#
# This allows numerical integration using standard ODE solvers like scipy.integrate.odeint.

def pendulum_derivs(state, t, g, L, beta, A, alpha):
    '''Calculate the derivatives for the pendulum motion.
    Inputs:
    state: Current state vector [theta, omega], 1D numpy array of length 2.
    t: Current time, float.
    g: Acceleration due to gravity, float.
    L: Length of the pendulum, float.
    beta: Damping coefficient, float.
    A: Amplitude of driving force, float.
    alpha: Frequency of driving force, float.
    Outputs:
    state_matrix: Derivatives [dtheta_dt, domega_dt], 1D numpy array of length 2.
    '''
    theta, omega = state
    
    # dθ/dt = ω
    dtheta_dt = omega
    
    # dω/dt = -(g/L)*sin(θ) - β*ω + (A/L)*cos(α*t)
    # The three terms represent:
    # -(g/L)*sin(theta): gravitational restoring force (nonlinear)
    # -beta*omega: damping force proportional to angular velocity
    # (A/L)*cos(alpha*t): external driving force (normalized by pendulum length)
    domega_dt = -(g/L) * np.sin(theta) - beta * omega + (A/L) * np.cos(alpha * t)
    
    state_matrix = np.array([dtheta_dt, domega_dt])
    
    return state_matrix


# Background: The Runge-Kutta 4th-order (RK4) method is a widely-used explicit numerical
# integration scheme for solving ordinary differential equations (ODEs). Given an ODE of the form:
# dy/dt = f(y, t), the RK4 method approximates the solution by computing a weighted average of
# four slope estimates (k1, k2, k3, k4) at different points within a time step.
#
# The RK4 algorithm for a single step from time t to t+dt is:
# k1 = f(y, t)                              [slope at the beginning]
# k2 = f(y + (dt/2)*k1, t + dt/2)           [slope at midpoint, using k1]
# k3 = f(y + (dt/2)*k2, t + dt/2)           [slope at midpoint, using k2]
# k4 = f(y + dt*k3, t + dt)                 [slope at the end, using k3]
# y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
#
# The weights (1, 2, 2, 1) and division by 6 give a fourth-order accurate approximation,
# meaning the local truncation error is O(dt^5) and the global error is O(dt^4).
# This method is particularly effective for smooth problems and provides good accuracy
# with moderate computational cost.

def runge_kutta_4th_order(f, state, t0, dt, n, g, L, beta, A, alpha):
    '''Run the RK4 integrator to solve the pendulum motion.
    Inputs:
    f: Derivatives function, which in general can be any ODE. In this context, it is defined as the pendulum_derivs function.
    state: Initial state vector [theta(t0), omega(t0)], 1D numpy array of length 2.
    t0: Initial time, float.
    dt: Time step, float.
    n: Number of steps, int.
    g: Acceleration due to gravity, float.
    L: Length of the pendulum, float.
    beta: Damping coefficient, float.
    A: Amplitude of driving force, float.
    alpha: Frequency of driving force, float.
    Outputs:
    trajectory: Array of state vectors, 2D numpy array of shape (n+1, 2).
    '''
    
    # Initialize the trajectory array with shape (n+1, 2) to store all states
    trajectory = np.zeros((n + 1, 2))
    
    # Store the initial state
    trajectory[0] = state.copy()
    
    # Current state and time
    y = state.copy()
    t = t0
    
    # Perform n integration steps
    for i in range(n):
        # Compute k1: slope at the current point (t, y)
        k1 = f(y, t, g, L, beta, A, alpha)
        
        # Compute k2: slope at the midpoint using k1
        k2 = f(y + (dt / 2.0) * k1, t + dt / 2.0, g, L, beta, A, alpha)
        
        # Compute k3: slope at the midpoint using k2
        k3 = f(y + (dt / 2.0) * k2, t + dt / 2.0, g, L, beta, A, alpha)
        
        # Compute k4: slope at the end of the interval using k3
        k4 = f(y + dt * k3, t + dt, g, L, beta, A, alpha)
        
        # Update state: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Update time
        t = t + dt
        
        # Store the new state in the trajectory
        trajectory[i + 1] = y.copy()
    
    return trajectory



# Background: The Global Truncation Error (GTE) is a measure of how much the numerical solution
# deviates from the true analytical solution. The step-doubling method estimates GTE by comparing
# two solutions computed with different timesteps:
# - One solution is computed with timestep dt
# - Another solution is computed with timestep dt/2
# The difference between these solutions provides an estimate of the error. For RK4, the error
# estimation uses the relationship:
# GTE ≈ |y(dt) - y(dt/2)| / 15
# This formula comes from Richardson extrapolation and the O(dt^4) accuracy of RK4.
#
# To find the optimized timestep, we sweep through a range of timesteps and evaluate each one
# using a combined metric: Metric = GTE × sqrt(Time). This metric balances accuracy (low GTE)
# and computational efficiency (low Time), with the square root applied to Time to avoid heavily
# penalizing slightly longer computation times while still prioritizing faster solutions.
#
# The algorithm:
# 1. Generate a range of timesteps logarithmically spaced between min_dt and max_dt
# 2. For each timestep, compute trajectories with dt and dt/2
# 3. Estimate GTE as the maximum difference between the two trajectories
# 4. Calculate the combined metric for each timestep
# 5. Select the timestep with the lowest combined metric
# 6. Return the trajectory computed with the optimized timestep

def pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps):
    '''Analyze a damped, driven pendulum system under various conditions.
    Inputs:
    g: Acceleration due to gravity, float.
    L: Length of the pendulum, float.
    beta: Damping coefficient, float.
    A: Amplitude of driving force, float.
    alpha: Frequency of driving force, float.
    initial_state: Initial state vector [theta, omega], list of floats.
    t0: Initial time, float.
    tf: Final time, float.
    min_dt: Smallest timestep, float.
    max_dt: Largest timestep, float.
    num_timesteps: Number of timesteps to generate, int.
    Outputs:
    optimized_trajectory: numpy array of floats, The trajectory of the pendulum system using the optimized timestep.
    '''
    
    # Convert initial state to numpy array
    state = np.array(initial_state, dtype=float)
    
    # Generate logarithmically spaced timesteps
    dt_values = np.logspace(np.log10(min_dt), np.log10(max_dt), num_timesteps)
    
    best_metric = float('inf')
    best_dt = min_dt
    
    # Sweep through different timesteps
    for dt in dt_values:
        # Calculate number of steps for this timestep
        n_steps = int(np.ceil((tf - t0) / dt))
        
        # Time the computation with dt
        start_time = time.time()
        trajectory_dt = runge_kutta_4th_order(pendulum_derivs, state, t0, dt, n_steps, 
                                               g, L, beta, A, alpha)
        time_dt = time.time() - start_time
        
        # Time the computation with dt/2
        start_time = time.time()
        trajectory_dt_half = runge_kutta_4th_order(pendulum_derivs, state, t0, dt / 2.0, 
                                                     n_steps * 2, g, L, beta, A, alpha)
        time_dt_half = time.time() - start_time
        
        # Estimate Global Truncation Error using step-doubling method
        # Only compare at common time points (every other point in the finer trajectory)
        gte = np.max(np.abs(trajectory_dt - trajectory_dt_half[::2])) / 15.0
        
        # Use the average time for metric calculation (approximate computational cost)
        avg_time = (time_dt + time_dt_half) / 2.0
        
        # Combined metric: GTE × sqrt(Time)
        # Add a small epsilon to avoid log(0) issues in case of very fast computations
        metric = gte * np.sqrt(avg_time + 1e-9)
        
        # Update best timestep if this metric is better
        if metric < best_metric:
            best_metric = metric
            best_dt = dt
    
    # Compute the final trajectory using the optimized timestep
    n_steps_optimal = int(np.ceil((tf - t0) / best_dt))
    optimized_trajectory = runge_kutta_4th_order(pendulum_derivs, state, t0, best_dt, 
                                                  n_steps_optimal, g, L, beta, A, alpha)
    
    return optimized_trajectory


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('78.3', 3)
target = targets[0]

g = 9.81
L = 0.1
beta = 0.1
A = 0.0
alpha = 0.0
initial_state = [0.1, 0.0]
t0 = 0.0
tf = 10.0
min_dt = 0.001
max_dt = 10
num_timesteps = 2
assert np.allclose(pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps), target)
target = targets[1]

g = 9.81
L = 0.5
beta = 0.2
A = 1.0
alpha = 0.5
initial_state = [0.5, 0.1]
t0 = 0.0
tf = 20.0
min_dt = 0.001
max_dt = 10
num_timesteps = 2
assert np.allclose(pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps), target)
target = targets[2]

g = 9.81
L = 0.3
beta = 1.0
A = 0.2
alpha = 1.0
initial_state = [1.0, 0.0]
t0 = 0.0
tf = 15.0
min_dt = 0.001
max_dt = 10
num_timesteps = 2
assert np.allclose(pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps), target)
