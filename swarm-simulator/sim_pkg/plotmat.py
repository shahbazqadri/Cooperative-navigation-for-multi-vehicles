import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('estimate_history.mat')

estimate_history = data['estimate_history']

print("estimate_history shape:", estimate_history.shape)

num_agents = estimate_history.shape[0]
num_time_steps = estimate_history.shape[2]

landmark_positions = np.array([[2., -5.], [10., 5.]])

# Create the plot
plt.figure(figsize=(10, 8))

# Plot agent trajectories
for agent_idx in range(num_agents):
    x_positions = estimate_history[agent_idx, 0, :]  # X coordinates
    y_positions = estimate_history[agent_idx, 1, :]  # Y coordinates

    # Plot the trajectory
    plt.plot(x_positions, y_positions, label=f'Agent {agent_idx + 1}')

    # Mark the starting and ending positions
    plt.plot(x_positions[0], y_positions[0], 'o', markersize=8)
    plt.plot(x_positions[-1], y_positions[-1], 'X', markersize=8)

# Plot landmarks as squares
for idx, (x, y) in enumerate(landmark_positions):
    plt.plot(x, y, 's', markersize=10, color='red', label='Landmark' if idx == 0 else "")

# Customize the plot
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Trajectories with Landmarks')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Equal scaling for x and y axes
plt.show()
