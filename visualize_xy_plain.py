import numpy as np
import matplotlib.pyplot as plt

# Function to read and visualize the 12xn data from a .txt file
def visualize_trajectory_from_txt(file_path, gt_file_path):
    # Load the data from the text file
    # data = np.loadtxt(file_path)
    data = np.loadtxt(file_path, delimiter=' ')  # For tab-separated values
    gt_data = np.loadtxt(gt_file_path, delimiter=' ')  # For tab-separated values
    
    print(data.shape)
    # Ensure the data has 12 columns (for 12xn)
    if data.shape[1] != 12:
        print(data.shape[1])
        raise ValueError("The data file should contain 12 columns")

    # Extract x, y, z coordinates (which are the 4th, 8th, and 12th columns)
    x_coords = data[:, 3]
    y_coords = data[:, 7]
    z_coords = data[:, 11]

    x_gt = gt_data[:, 3]
    y_gt = gt_data[:, 7]
    z_gt = gt_data[:, 11]


    # Plotting the trajectory in 3D space
    fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    # ax.plot(x_coords, y_coords, z_coords, marker='o')
    ax.plot(x_coords, y_coords)
    ax.plot(x_gt, y_gt)

    # Labels for the plot
    ax.set_title('3D Trajectory Visualization')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')

    # Display the plot
    plt.show()

# Example usage
file_path = '/home/leedk/deepclr/scenario/20241002_170232_etri_00-05_DEEPCLRTF/kitti/00.txt'  
gt_file_path = '/home/leedk/deepclr/scenario/20241002_170232_etri_00-05_DEEPCLRTF/kitti/00_gt.txt'  # Replace with the path to your .txt file
visualize_trajectory_from_txt(file_path, gt_file_path)

