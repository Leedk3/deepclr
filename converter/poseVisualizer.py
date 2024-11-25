import os
import numpy as np
from math import cos, sin, radians, degrees
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_transform(line):
    """Parses a line with Transform information."""
    parts = line.strip().replace('Transform(Location(', '').replace('), Rotation(', ',').replace('))', '').split(',')
    x = float(parts[0].split('=')[1])
    y = float(parts[1].split('=')[1])
    z = float(parts[2].split('=')[1])
    pitch = float(parts[3].split('=')[1])
    yaw = float(parts[4].split('=')[1])
    roll = float(parts[5].split('=')[1])
    return np.array([x, y, z]), radians(pitch), radians(yaw), radians(roll)

def compute_kitti_transform(location, pitch, yaw, roll):
    """Computes the KITTI-style transform matrix."""
    R_x = np.array([[1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll), cos(roll)]])
    
    R_y = np.array([[cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [-sin(pitch), 0, cos(pitch)]])
    
    R_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw), cos(yaw), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = location
    
    kitti_transform = T[:3, :].reshape(-1)
    return kitti_transform

def visualize_poses(positions, orientations):

    """Visualizes the positions in 3D space."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot positions
    positions = np.array(positions)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', label="Positions")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Position Plot')

    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.legend()
    plt.show()

    """Visualizes the positions and orientations on a 3D plot."""
    fig2 = plt.figure(figsize=(10, 8))
    
    # Subplot for Roll
    ax1 = fig2.add_subplot(311)
    ax1.plot(range(len(positions)), [x for x, y, z in positions], 'r-', label='X')
    ax1.set_ylabel('X')
    ax1.set_title('X, Y, Z over Time')
    ax1.grid(True)
    
    # Subplot for Pitch
    ax2 = fig2.add_subplot(312)
    ax2.plot(range(len(positions)), [y for x, y, z in positions], 'g-', label='Y')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    
    # Subplot for Yaw
    ax3 = fig2.add_subplot(313)
    ax3.plot(range(len(positions)), [z for x, y, z in positions], 'b-', label='Z')
    ax3.set_ylabel('Z')
    ax3.set_xlabel('Index')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

    """Visualizes the positions and orientations on a 3D plot."""
    fig = plt.figure(figsize=(10, 8))
    
    # Subplot for Roll
    ax1 = fig.add_subplot(311)
    ax1.plot(range(len(orientations)), [degrees(roll) for roll, pitch, yaw in orientations], 'r-', label='Roll')
    ax1.set_ylabel('Roll (degrees)')
    ax1.set_title('Roll, Pitch, Yaw over Time')
    ax1.grid(True)
    
    # Subplot for Pitch
    ax2 = fig.add_subplot(312)
    ax2.plot(range(len(orientations)), [degrees(pitch) for roll, pitch, yaw in orientations], 'g-', label='Pitch')
    ax2.set_ylabel('Pitch (degrees)')
    ax2.grid(True)
    
    # Subplot for Yaw
    ax3 = fig.add_subplot(313)
    ax3.plot(range(len(orientations)), [degrees(yaw) for roll, pitch, yaw in orientations], 'b-', label='Yaw')
    ax3.set_ylabel('Yaw (degrees)')
    ax3.set_xlabel('Index')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def process_files(input_dir, output_dir, sub_dir):
    """Processes each .txt file and converts to KITTI format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pose_dir = os.path.join(input_dir, sub_dir, 'coords')
    scan_dir = os.path.join(input_dir, sub_dir, 'lidar')
    
    output_file_path = os.path.join(output_dir, 'poses', f'{sub_dir}.txt')
    output_scan_path = os.path.join(output_dir, 'sequences', sub_dir, 'velodyne')
    
    output_time_path = os.path.join(output_dir, 'sequences', sub_dir, 'times.txt')
    
    output_file_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    if not os.path.exists(output_scan_path):
        os.makedirs(output_scan_path)

    poses_files = sorted([f for f in os.listdir(pose_dir) if os.path.isfile(os.path.join(pose_dir, f)) and f.endswith('.txt')])
    scan_files = sorted([f for f in os.listdir(scan_dir) if os.path.isfile(os.path.join(scan_dir, f)) and f.endswith('.npy')])

    time_start = 0
    time_step = 0.1

    positions = []
    orientations = []
    
    with open(output_time_path, 'w') as time_file:  # Open the output file once for appending
        with open(output_file_path, 'w') as out_file:  # Open the output file once for appending
            for pose_name_buf, scan_name_buf in zip(poses_files, scan_files):
                print(pose_name_buf, scan_name_buf)
                input_path = os.path.join(pose_dir, pose_name_buf)

                with open(input_path, 'r') as file:
                    line = file.readline()
                    location, pitch, yaw, roll = parse_transform(line)
                    kitti_transform = compute_kitti_transform(location, pitch, yaw, roll)
                    print(location[2], kitti_transform)
                    # Store positions and orientations
                    positions.append(location)
                    orientations.append((roll, pitch, yaw))
                    
                    # Write the transformation matrix to the output file
                    # out_file.write(" ".join([f"{num:.6e}" for num in kitti_transform]) + "\n")
                    # time_file.write(f"{time_start:.6e}" + "\n")
                    
                # if scan_name_buf.endswith('.npy'):
                #     npy_file = os.path.join(scan_dir, scan_name_buf)
                #     bin_file = os.path.join(output_scan_path, scan_name_buf.replace('.npy', '.bin'))
                #     npy_to_bin(npy_file, bin_file)
                #     print(f"Converted {npy_file} to {bin_file}")
                
                time_start = time_start + time_step
    
    # Visualize the roll, pitch, yaw over time (index)
    visualize_poses(positions, orientations)
        
# Example usage
input_directory = '/home/leedk/Dataset/etri/'  # Replace with your input directory
output_directory = '/home/leedk/Dataset/etri/dataset'  # Replace with your output directory
sub_directory = 'town10'

process_files(input_directory, output_directory, sub_directory)
