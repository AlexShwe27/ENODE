import time
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, cos, Matrix

import torch
import torch.optim as optim

from torchdiffeq import odeint

import CartPole_neural_networks as net

# data parameters
data_size = 10000
batch_size = 750
batch_time = 10

# Define the cart-pole model
N = 2   # Degrees of freedom
l = 1   # Kinematic parameters

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


# Simulate the pendulum for multiple initial conditions
init_cond = [120.0,     0.0,    0.0,   0.0]
init_cond = torch.tensor([np.deg2rad(init_cond[0]), init_cond[1], init_cond[2], init_cond[3]]).float()

# Time parameters
nsteps = data_size
t = torch.linspace(0.0, 30.0, nsteps)

# Run the simulation for each initial condition
true_trajectory = odeint(cartpole_true_trajectory, init_cond, t)

# Add noise to the true trajectory
noise_std = 0.0001
noise = noise_std*torch.randn_like(true_trajectory)
true_trajectory = true_trajectory+noise


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


def get_batch():
  s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))

  batch_y0 = true_trajectory[s]     # (M, D)
  batch_t = t[:batch_time]  # (T)
  batch_y = torch.stack([true_trajectory[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)

  return batch_y0, batch_t, batch_y


ii = 0
test_freq=20
niters=3000

kinetic_neural_network = net.kinetic_net()
kinetic_optimizer = optim.Adam(kinetic_neural_network.parameters(), lr=0.02)
kinetic_scheduler = optim.lr_scheduler.StepLR(kinetic_optimizer, step_size=500, gamma=0.5)

potential_neural_network = net.potential_net()
potential_optimizer = optim.Adam(potential_neural_network.parameters(), lr=0.02)
potential_scheduler = optim.lr_scheduler.StepLR(potential_optimizer, step_size=500, gamma=0.5)

loss_function = nn.MSELoss()
end = time.time()

loss_values = []
val_loss_values = []

for itr in range(1, niters + 1):

    batch_y0, batch_t, batch_y = get_batch()
    pred_y = odeint(cartpole_energy_neuralODE, batch_y0, batch_t, method="rk4")
    loss = loss_function(pred_y, batch_y)

    kinetic_optimizer.zero_grad()
    potential_optimizer.zero_grad()

    loss.backward()

    kinetic_optimizer.step()
    potential_optimizer.step()
    kinetic_scheduler.step()
    potential_scheduler.step()

    loss_values.append(loss.item())

    print('Training Process : {:.2f}% | Train Loss : {:.12f}'.format(itr / niters * 100, loss.item()))

    end = time.time()

# Save the trained model
torch.save(kinetic_neural_network.state_dict(), "cartpole_grad_kinetic.pth")
torch.save(potential_neural_network.state_dict(), "cartpole_grad_potential.pth")

pred_trajectory = odeint(cartpole_energy_neuralODE, init_cond, t, method='rk4')
true_trajectory = true_trajectory.detach().numpy()
pred_trajectory = pred_trajectory.detach().numpy()

plt.figure(1)
plt.semilogy(loss_values)
plt.semilogy(val_loss_values)
plt.semilogy()
plt.show()

plt.figure(2, figsize=(10, 10))

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
