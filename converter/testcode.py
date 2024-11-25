import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Set the directory containing the .npy files
directory = '/home/leedk/Dataset/etri/town03/lidar'  # Replace with your directory path

# Initialize a list to hold the data
data_list = []

# Loop over the file numbers from 000000 to 000450
for i in range(100):  # 451 to include 000450.npy
    file_name = f"{i:06d}.npy"  # Format the number as 6 digits, with leading zeros
    file_path = os.path.join(directory, file_name)
    
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(len(data))
        print(data)
        data_list.append(data)
    else:
        print(f"File {file_name} does not exist.")
        


# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the scatter plot
scat = ax.scatter([], [], [], s=5)

# Set axis limits
ax.set_xlim([-40, 40])
ax.set_ylim([-40, 40])
ax.set_zlim([-10, 40])

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Update function for animation
def update(frame):
    # Clear the current scatter plot
    ax.cla()
    
    # Set axis limits (to maintain during clearing)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([-10, 40])
    
    # Plot the new data
    ax.scatter(data_list[frame][:, 0], data_list[frame][:, 1], data_list[frame][:, 2], s=5)
    
    # Set axis labels (after clearing)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data_list), interval=100)

# Show the animation
plt.show()