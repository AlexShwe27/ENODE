import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation

# Parameters
mc = 5.0    # Mass of the cart
mp = 2.0    # Mass of the pole
l = 0.75    # Length of the pole
g = 9.8     # Gravitational constant
k = 0.5     # Spring constant


# Define the ODE for the cart-pole system
def cartpole_true_trajectory(t, state, gamma=0.02):
    x, theta, px, ptheta = state[0], state[1], state[2], state[3]
    dstate = torch.zeros_like(state)

    dstate[0] = ((mc + mp) / (mc ** 2)) * px - (1 / mc) * l * ptheta * torch.cos(theta)
    dstate[1] = (-1 / mc) * l * px * torch.cos(theta) + (1 / mp) * (l ** 2) * ptheta
    dstate[2] = -k*x - gamma*px
    dstate[3] = (-1 / mc) * l * px * ptheta * torch.sin(theta) - mp * g * l * torch.sin(theta) - gamma*ptheta

    return dstate


# Initial conditions: [initial x, initial theta (in degrees), initial px, initial ptheta]
init_cond = [0.0, 120.0, 0.0, 0.0]
init_cond = torch.tensor([init_cond[0], np.deg2rad(init_cond[1]), init_cond[2], init_cond[3]]).float()

# Simulation time
data_size = 4000  # Reduce data size for animation speed
t = torch.linspace(0.0, 75.0, data_size)  # Simulate for 50 seconds with 2500 time steps

# Solve the ODE using torchdiffeq's odeint
true_trajectory = odeint(cartpole_true_trajectory, init_cond, t, method="rk4")

# Extract the results
x = true_trajectory[:, 0].detach().numpy()       # Cart position over time
theta = true_trajectory[:, 1].detach().numpy()   # Pole angle over time

# Convert theta to Cartesian coordinates for the pole’s end point
pole_x = x + l * np.sin(theta)   # x position of the pole's end
pole_y = -l * np.cos(theta)      # y position of the pole's end

# Set up the figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

# Create the cart as a larger rectangle
cart_width = 0.4  # Increased width of the cart
cart_height = 0.2  # Increased height of the cart
cart = Rectangle((x[0] - cart_width / 2, -cart_height / 2), cart_width, cart_height, color='black')
ax.add_patch(cart)

# Create the pole as a line object
pole, = ax.plot([], [], 'k-', lw=3)  # Pole as a line

# Create the pendulum as a small circle
pendulum_radius = 0.05  # Radius of the pendulum circle
pendulum_circle = Circle((pole_x[0], pole_y[0]), pendulum_radius, color='black')
ax.add_patch(pendulum_circle)

# Initialize path line for the pendulum's trace
path_line, = ax.plot([], [], 'b--', lw=1)  # Blue dashed line to represent the path

# Initialize lists to store the trace
trace_history = 2000  # Number of points to keep in the trace
trace_x = []
trace_y = []
trace_patches = [ax.plot([], [], 'o', markersize=4, color='red', alpha=1 - i / trace_history)[0] for i in range(trace_history)]

# Initialize the animation
def init():
    cart.set_xy((x[0] - cart_width / 2, -cart_height / 2))
    pole.set_data([], [])
    pendulum_circle.set_center((pole_x[0], pole_y[0]))
    path_line.set_data([], [])
    for trace in trace_patches:
        trace.set_data([], [])
    return cart, pole, pendulum_circle, path_line, *trace_patches

# Update function for the animation
def update(i):
    # Update cart position as a rectangle
    cart.set_xy((x[i] - cart_width / 2, -cart_height / 2))

    # Update pole position as a line segment from the cart to the pole end
    pole.set_data([x[i], pole_x[i]], [0, pole_y[i]])

    # Update the pendulum circle position to be at the end of the pole
    pendulum_circle.set_center((pole_x[i], pole_y[i]))

    # Append the current pendulum position to the trace
    trace_x.append(pole_x[i])
    trace_y.append(pole_y[i])

    # Keep the last `trace_history` points in the trace
    if len(trace_x) > trace_history:
        trace_x.pop(0)
        trace_y.pop(0)

    # Update the trace patches with fading effect
    for j, trace in enumerate(trace_patches):
        if j < len(trace_x):
            trace.set_data([trace_x[-(j+1)]], [trace_y[-(j+1)]])  # Update each point with a slight fade

    return cart, pole, pendulum_circle, path_line, *trace_patches

# Create the animation, set repeat=False to stop at the end
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20, repeat=False)

# Display the animation
plt.title("Inverted Cart-Pole with Fading Path Trace")
plt.xlabel("x position")
plt.ylabel("y position")
plt.show()  # No plt.grid() to keep the plot clear of grid lines
