import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# def calculate_relative_errors(data, gt_data):
#     """
#     Calculate relative position error (RPE) and relative rotational error.
#     Arguments:
#     - data: Estimated trajectory (Nx12 matrix)
#     - gt_data: Ground truth trajectory (Nx12 matrix)
    
#     Returns:
#     - rpe: Relative Position Errors
#     - rot_error: Relative Rotational Errors
#     """
#     num_frames = data.shape[0] - 1
#     rpe = []
#     rot_error = []

#     for i in range(num_frames):
#         # Calculate relative pose transformation for data
#         T1_est = np.reshape(data[i], (3, 4))
#         T2_est = np.reshape(data[i + 1], (3, 4))
#         rel_est = np.dot(np.linalg.inv(T1_est[:, :3]), T2_est[:, :3])  # Relative transformation

#         # Calculate relative pose transformation for ground truth
#         T1_gt = np.reshape(gt_data[i], (3, 4))
#         T2_gt = np.reshape(gt_data[i + 1], (3, 4))
#         rel_gt = np.dot(np.linalg.inv(T1_gt[:, :3]), T2_gt[:, :3])  # Relative transformation

#         # Compute position errors (translation difference)
#         trans_est = T2_est[:, 3] - T1_est[:, 3]
#         trans_gt = T2_gt[:, 3] - T1_gt[:, 3]
#         trans_error = np.linalg.norm(trans_est - trans_gt)

#         # Compute rotational errors
#         rel_rotation_error = np.linalg.norm(np.arccos(np.clip(np.trace(np.dot(rel_gt.T, rel_est)) / 2.0 - 1.0, -1.0, 1.0)))

#         rpe.append(trans_error)
#         rot_error.append(rel_rotation_error)

#     return np.array(rpe), np.array(rot_error)

def calculate_relative_errors(data, gt_data):
    """
    Calculate relative position error (RPE) and relative rotational error using geodesic distance.
    Arguments:
    - data: Estimated trajectory (Nx12 matrix)
    - gt_data: Ground truth trajectory (Nx12 matrix)
    
    Returns:
    - rpe: Relative Position Errors
    - rot_error: Relative Rotational Errors
    """
    num_frames = data.shape[0] - 1
    rpe = []
    rot_error = []

    for i in range(num_frames):
        # Extract rotation and translation for estimated data
        T1_est = np.reshape(data[i], (3, 4))
        T2_est = np.reshape(data[i + 1], (3, 4))
        rel_est_rot = np.dot(np.linalg.inv(T1_est[:, :3]), T2_est[:, :3])

        # Extract rotation and translation for ground truth data
        T1_gt = np.reshape(gt_data[i], (3, 4))
        T2_gt = np.reshape(gt_data[i + 1], (3, 4))
        rel_gt_rot = np.dot(np.linalg.inv(T1_gt[:, :3]), T2_gt[:, :3])

        # Compute position errors (translation difference)
        trans_est = T2_est[:, 3] - T1_est[:, 3]
        trans_gt = T2_gt[:, 3] - T1_gt[:, 3]
        trans_error = np.linalg.norm(trans_est - trans_gt)

        # Compute rotational errors using geodesic distance
        try:
            rel_error_matrix = np.dot(rel_gt_rot.T, rel_est_rot)
            rotation_distance = R.from_matrix(rel_error_matrix).magnitude()
        except ValueError:
            rotation_distance = np.pi  # Assign max error if rotation is invalid

        rpe.append(trans_error)
        rot_error.append(rotation_distance)

    return np.array(rpe), np.array(rot_error)

def calculate_absolute_errors(data, gt_data):
    """
    Calculate absolute position error (APE).
    Arguments:
    - data: Estimated trajectory (Nx12 matrix)
    - gt_data: Ground truth trajectory (Nx12 matrix)
    
    Returns:
    - ape: Absolute Position Errors
    """
    # Extract position vectors (last column of the transformation matrix for each frame)
    positions_est = data[:, [3, 7, 11]]  # x, y, z columns
    positions_gt = gt_data[:, [3, 7, 11]]  # x, y, z columns

    # Compute Euclidean distance between estimated and ground truth positions
    ape = np.linalg.norm(positions_est - positions_gt, axis=1)

    return ape

def plot_relative_errors(rpe, rot_error):
    """
    Plots the relative position error (RPE) and relative rotational error.
    Arguments:
    - rpe: Array of relative position errors
    - rot_error: Array of relative rotational errors
    """
    # Convert rotational errors to degrees
    rot_error_deg = np.degrees(rot_error)

    # Plot RPE
    fig2 = plt.figure(figsize=(12, 6))
    plt.rc('font', size=18)        # 기본 폰트 크기
    plt.rc('legend', fontsize=18)  # 범례 폰트 크기
    plt.plot(rpe, label='Relative Position Error (RPE)', marker='o')
    # plt.title('Relative Position Error (RPE)')
    plt.xlabel('Frame Index')
    plt.ylabel('RPE [meters]')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot rotational error
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(rot_error_deg, label='Relative Rotational Error', marker='x', color='orange')
    # plt.title('Relative Rotational Error')
    plt.xlabel('Frame Index')
    plt.ylabel('Rotational Error [degrees]')
    plt.legend()
    plt.grid()
    plt.rc('font', size=18)        # 기본 폰트 크기
    plt.rc('legend', fontsize=18)  # 범례 폰트 크기
    plt.show()

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


    ape = calculate_absolute_errors(data, gt_data)

    # Calculate RPE and rotational errors
    rpe, rot_error = calculate_relative_errors(data, gt_data)

    # Plot relative errors
    plot_relative_errors(rpe, rot_error)
    
    # Print error statistics
    print("Absolute Position Errors (APE):")
    print(f"Mean: {np.mean(ape):.4f}, Std Dev: {np.std(ape):.4f}")
    # Print error statistics
    print("Relative Position Errors (RPE):")
    print(f"Mean: {np.mean(rpe):.4f}, Std Dev: {np.std(rpe):.4f}")
    print("Relative Rotational Errors (degrees):")
    print(f"Mean: {np.mean(np.degrees(rot_error)):.4f}, Std Dev: {np.std(np.degrees(rot_error)):.4f}")


    plt.rc('font', size=18)        # 기본 폰트 크기
    plt.rc('legend', fontsize=18)  # 범례 폰트 크기

    # Plotting the trajectory in 3D space
    fig1 = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig1.add_subplot(111)
    # ax.plot(x_coords, y_coords, z_coords, marker='o')
    ax.plot(x_coords, y_coords, label='Estimated')
    ax.plot(x_gt, y_gt, label='Ground Truth')

    # Labels for the plot
    # ax.set_title('3D Trajectory Visualization')
    ax.set_xlabel('X [meter]')
    ax.set_ylabel('Y [meter]')
    # ax.set_zlabel('Z axis')

    # Display the plot
    plt.axis('equal')
    plt.legend()
    plt.rc('font', size=18)        # 기본 폰트 크기
    plt.rc('legend', fontsize=18)  # 범례 폰트 크기
    plt.grid()
    plt.show()

# Example usage
# file_path = '/home/leedk/deepclr/scenario/20241002_170232_etri_00-05_DEEPCLRTF/kitti/05.txt'  
# gt_file_path = '/home/leedk/deepclr/scenario/20241002_170232_etri_00-05_DEEPCLRTF/kitti/05_gt.txt'  # Replace with the path to your .txt file

file_path = '/home/leedk/deepclr/scenario/20241002_174235_kitti_07-10_DEEPCLRTF/kitti/07.txt'  
gt_file_path = '/home/leedk/deepclr/scenario/20241002_174235_kitti_07-10_DEEPCLRTF/kitti/07_gt.txt'  # Replace with the path to your .txt file

visualize_trajectory_from_txt(file_path, gt_file_path)
