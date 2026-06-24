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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('78.1', 3)
target = targets[0]

def test_case_1():
    state = np.array([0.1, 0.0])  # initial theta and omega
    t = 0.0  # initial time
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.0  # no damping
    A = 0.0  # no driving force
    alpha = 0.0  # driving frequency (irrelevant since A=0)
    expected_output = np.array([0.0, - (g / L) * np.sin(0.1)])  # [dtheta_dt, domega_dt]
    output = pendulum_derivs(state, t, g, L, beta, A, alpha)
    assert np.allclose(output, expected_output), "Test Case 1 Failed"
# Run test cases
test_case_1()
target = targets[1]

def test_case_2():
    state = np.array([0.1, 0.2])  # initial theta and omega
    t = 0.0  # initial time
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.1  # damping coefficient
    A = 0.0  # no driving force
    alpha = 0.0  # driving frequency (irrelevant since A=0)
    expected_output = np.array([0.2, - (g / L) * np.sin(0.1) - 0.1 * 0.2])  # [dtheta_dt, domega_dt]
    output = pendulum_derivs(state, t, g, L, beta, A, alpha)
    assert np.allclose(output, expected_output), "Test Case 2 Failed"
test_case_2()
target = targets[2]

def test_case_3():
    state = np.array([0.1, 0.2])  # initial theta and omega
    t = 1.0  # time
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.1  # damping coefficient
    A = 1.0  # amplitude of driving force
    alpha = 2.0  # driving frequency
    expected_output = np.array([0.2, - (g / L) * np.sin(0.1) - 0.1 * 0.2 + 1.0 * np.cos(2.0 * 1.0)])  # [dtheta_dt, domega_dt]
    output = pendulum_derivs(state, t, g, L, beta, A, alpha)
    assert np.allclose(output, expected_output), "Test Case 3 Failed"
test_case_3()
