import os
import numpy as np
from math import cos, sin, radians

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

def npy_to_bin(npy_file, bin_file):
    """Converts a .npy file to KITTI-style .bin file."""
    data = np.load(npy_file)
    if data.shape[1] != 4:
        raise ValueError("The input data must have four columns: [x, y, z, intensity].")
    data.astype('float32').tofile(bin_file)

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
    
    with open(output_time_path, 'w') as time_file:  # Open the output file once for appending
            
        with open(output_file_path, 'w') as out_file:  # Open the output file once for appending
            for pose_name_buf, scan_name_buf in zip(poses_files, scan_files):
                print(pose_name_buf, scan_name_buf)
                input_path = os.path.join(pose_dir, pose_name_buf)

                with open(input_path, 'r') as file:
                    line = file.readline()
                    location, pitch, yaw, roll = parse_transform(line)
                    kitti_transform = compute_kitti_transform(location, pitch, yaw, roll)
                    
                    # Write the transformation matrix to the output file
                    out_file.write(" ".join([f"{num:.6e}" for num in kitti_transform]) + "\n")
                    time_file.write(f"{time_start:.6e}" + "\n")
                    

                if scan_name_buf.endswith('.npy'):
                    npy_file = os.path.join(scan_dir, scan_name_buf)
                    bin_file = os.path.join(output_scan_path, scan_name_buf.replace('.npy', '.bin'))
                    npy_to_bin(npy_file, bin_file)
                    print(f"Converted {npy_file} to {bin_file}")
                
                time_start = time_start + time_step
        
# Example usage
input_directory = '/home/leedk/Dataset/etri/'  # Replace with your input directory
output_directory = '/home/leedk/Dataset/etri/dataset'  # Replace with your output directory
sub_directory = 'town10'

process_files(input_directory, output_directory, sub_directory)
