import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
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
init_cond = [90.0,     0.0,    0.0,   0.0]
init_cond = torch.tensor([np.deg2rad(init_cond[0]), init_cond[1], init_cond[2], init_cond[3]]).float()

# Simulation time
nsteps = data_size
t = torch.linspace(0.0, 30.0, nsteps)

true_trajectory = odeint(cartpole_true_trajectory, init_cond, t, method="rk4")



def cartpole_hamiltonian_neuralODE(t, state):

    state = state.requires_grad_(True)

    # Obtain Hamiltonian from Neural Network
    H = neural_network(state)

    # Calculate the gradient of Hamiltonian w.r.t. the coordinates
    dH = torch.autograd.grad(outputs=H, inputs=state, grad_outputs=torch.ones_like(H), create_graph=True)[0]

    if dH.dim() == 2:
        # Calculate dqdt and dpdt
        dqdt = dH[:, 2:4]
        dpdt = -dH[:, 0:2]
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.cat([dqdt, dpdt], dim=1)

    else:
        # Calculate dqdt and dpdt
        dqdt = dH[2:4]
        dpdt = -dH[0:2]
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

neural_network = net.hamiltonian_net()
optimizer = optim.Adam(neural_network.parameters(), lr=0.02)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)  # Decrease LR by half every 1000 epochs
loss_function = nn.MSELoss()
end = time.time()

loss_values = []
val_loss_values = []

for itr in range(1, niters + 1):

    batch_y0, batch_t, batch_y = get_batch()
    pred_y = odeint(cartpole_hamiltonian_neuralODE, batch_y0, batch_t, method="heun3")
    loss = loss_function(pred_y, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_values.append(loss.item())

    print('Training Process : {:.2f}% | Train Loss : {:.12f}'.format(itr/niters * 100, loss.item()))

    end = time.time()

# Save the trained model
torch.save(neural_network.state_dict(), "cartpole_hamiltonian.pth")

plt.semilogy(loss_values)
plt.semilogy(val_loss_values)
plt.semilogy()
plt.show()

