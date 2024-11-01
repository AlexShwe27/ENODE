import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sympy as sym
from sympy import symbols, cos, Matrix


from torchdiffeq import odeint as odeint

import CartPole_neural_networks as net

# Parameters
mc = 5.0
mp = 2.0
l = 0.75
g = 9.8
k = 0.5


# Define symbolic variables
q_sym = symbols('q0:2')
p_sym = symbols('p0:2')
p_sym_vec = Matrix(p_sym)

# Define symbolic mass matrix
M_sym = Matrix([[mp*l**2, mp*l*cos(q_sym[0])],
                [mp*l*cos(q_sym[0]), mc + mp]])
M_inv_sym = M_sym.inv()

# Define symbolic potential energy
V_sym = mp * g * l * cos(q_sym[0]) + 0.5*k*(q_sym[1]**2)

# Convert potential energy V_sym to a matrix for compatibility
V_sym_matrix = Matrix([[V_sym]])

# Define symbolic kinetic energy
T_sym = 0.5 * p_sym_vec.T * M_inv_sym * p_sym_vec

# Define symbolic Hamiltonian
H_sym = T_sym + V_sym_matrix

# Compute Jacobians
# Jacobian with respect to q_sym
H_q_jacobian = H_sym.jacobian(q_sym)

# Jacobian with respect to p_sym
H_p_jacobian = H_sym.jacobian(p_sym)

# Convert symbolic Jacobian to a numerical function
# Use lambdify to create a function for H_q_jacobian
H_q_jacobian_func = sym.lambdify((q_sym, p_sym), H_q_jacobian, 'numpy')
H_p_jacobian_func = sym.lambdify((q_sym, p_sym), H_p_jacobian, 'numpy')


# Define the cart-pole ODE
def cartpole_true_trajectory(t, state):
    q = state[0:2].numpy()
    p = state[2:].numpy()
    dqdt = torch.tensor(H_p_jacobian_func(q, p).flatten(), dtype=torch.float32)
    dpdt = torch.tensor(-H_q_jacobian_func(q, p).flatten(), dtype=torch.float32)
    return torch.cat([dqdt, dpdt])


def cartpole_energy_neuralODE(t, state):

    state = state.requires_grad_(True)

    if state.dim() == 2:
        q = state[:, 0:2]
        p = state[:, 2:4]
    else:
        q = state[0:2]
        p = state[2:4]

    # Obtain the gradients of the Hamiltonian from Neural Network
    T = kinetic_neural_network(state)
    V = potential_neural_network(q)

    dT = torch.autograd.grad(outputs=T, inputs=state, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dVdq = torch.autograd.grad(outputs=V, inputs=q, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    if state.dim() == 2:
        # Calculate dqdt and dpdt
        dqdt = dT[:, 2:4]
        dpdt = -dT[:, 0:2] - dVdq
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.cat([dqdt, dpdt], dim=1)

    else:
        # Calculate dqdt and dpdt
        dqdt = dT[2:4]
        dpdt = -dT[0:2] - dVdq
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.cat([dqdt, dpdt])

    return dstate


# data parameters
data_size = 2000

# Initial Conditions
init_cond = [120.0,     0.0,    0.0,   0.0]
init_cond = torch.tensor([np.deg2rad(init_cond[0]), init_cond[1], init_cond[2], init_cond[3]]).float()

# Simulation time
nsteps = data_size
t = torch.linspace(0.0, 30.0, nsteps)

kinetic_neural_network = net.kinetic_net()
kinetic_neural_network.load_state_dict(torch.load("cartpole_grad_kinetic.pth", weights_only=True))
kinetic_neural_network.eval()

potential_neural_network = net.potential_net()
potential_neural_network.load_state_dict(torch.load("cartpole_grad_potential.pth", weights_only=True))
potential_neural_network.eval()

# Solve the ODE using odeint
true_trajectory = odeint(cartpole_true_trajectory, init_cond, t, method='rk4')
pred_trajectory = odeint(cartpole_energy_neuralODE, init_cond, t, method='rk4')
true_trajectory = true_trajectory.detach().numpy()
pred_trajectory = pred_trajectory.detach().numpy()

# Plot the true and predicted trajectories
plt.figure(1, figsize=(10, 10))

# Plot the position of cart
plt.subplot(2, 2, 1)
plt.plot(t, true_trajectory[:, 0], label=r'true x')
plt.plot(t, pred_trajectory[:, 0], 'r--', label=r'approx x')
plt.title('Position of the cart(x)')
plt.xlabel('Time [s]')
plt.ylabel('Distance [m]')
plt.legend()

# Plot the angle of pendulum
plt.subplot(2, 2, 2)
plt.plot(t, true_trajectory[:, 1], label=r'true $\theta$')
plt.plot(t, pred_trajectory[:, 1], 'r--', label=r'approx $\theta$')
plt.title(r'Angle of the Pendulum ($\theta$)')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()

# Plot the linear momentum
plt.subplot(2, 2, 3)
plt.plot(t, true_trajectory[:, 2], label=r'$p_x$')
plt.plot(t, pred_trajectory[:, 2], 'r--', label=r'approx $p_x$')
plt.title(r'Linear Momentum of the Pendulum ($p_x$)')
plt.xlabel('Time [s]')
plt.ylabel('Linear Momentum [kg.m/s]')
plt.legend()

# Plot the angular momentum
plt.subplot(2, 2, 4)
plt.plot(t, true_trajectory[:, 3], label=r'$p_\theta$')
plt.plot(t, pred_trajectory[:, 3], 'r--', label=r'approx $p_\theta$')
plt.title('Angular Momentum of the Pendulum (x)')
plt.xlabel('Time [s]')
plt.ylabel('Angular Momentum [kg.m^2/s]')
plt.legend()


# Calculate the total Energy
def Energy(trajectory):

    x = trajectory[:, 0]
    theta = trajectory[:, 1]
    px = trajectory[:, 2]
    ptheta = trajectory[:, 3]

    # Calculate Kinetic Energy
    T = 0.5*((mc+mp)/(mc**2))*(px**2) - (1/mc)*l*px*ptheta*np.cos(theta) + 0.5*(1/mp)*(l**2)*(ptheta**2)
    # Calculate Potential Energy
    V = mp*g*(l-(l*np.cos(theta))) + 0.5*k*(x**2)

    return T, V


# Obtain true and predicted energy
true_T, true_V = Energy(true_trajectory)
pred_T, pred_V = Energy(pred_trajectory)

true_total_energy = true_T + true_V
pred_total_energy = pred_T + pred_V

# Observe the energy behaviours
plt.figure(2, figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(t, true_T, label='True T')
plt.plot(t, pred_T, 'r--', label='predicted T')
plt.title('Kinetic Energy (T)')
plt.xlabel('Time [s]')
plt.ylabel('Kinetic Energy [J]')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, true_V, label='True V')
plt.plot(t, pred_V, 'r--', label='predicted V')
plt.title('Potential Energy (V')
plt.xlabel('Time [s]')
plt.ylabel('Potential Energy [J]')
plt.legend()

plt.figure(3, figsize=(10, 5))
plt.plot(t, true_total_energy, label='True Total Energy')
plt.plot(t, pred_total_energy, 'r--', label='predicted Total Energy')
plt.title('Total Energy')
plt.xlabel('Time [s]')
plt.ylabel('Total Energy [J]')
plt.legend()

plt.show()