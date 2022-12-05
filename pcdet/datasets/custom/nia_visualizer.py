import copy
import pickle

import numpy as np

import json
import os
import glob

import shutil
import platform
import yaml
from pathlib import Path
from easydict import EasyDict
import sys
import shutil
import open3d
import math
import pandas as pd

class NiaVisualizer():
    def __init__(self, root_path = None):
        self.root = root_path

        print(self.root)
        self.points_dir = os.path.join(self.root, "points")
        self.labels_dir = os.path.join(self.root, "labels")
        self.debug_dir = os.path.join(self.root, "debug")

        try:
            for points_file_name, labels_file_name, debug_file_name in zip(sorted(os.listdir(self.points_dir)), sorted(os.listdir(self.labels_dir)), sorted(os.listdir(self.debug_dir))):
                points_file_abs = os.path.join(self.points_dir, points_file_name)    
                print(points_file_abs)      

                obj_points = np.fromfile(str(points_file_abs), dtype=np.float32)
                obj_points = obj_points.reshape(-1,4)

                debug_file_abs = os.path.join(self.debug_dir, debug_file_name)    

                point_cloud = open3d.geometry.PointCloud()
                xyz_points = obj_points[:, :3]
                xyz_points = np.delete(xyz_points, np.where((abs(xyz_points) >= 50.))[0], axis=0)

                # for xyz in xyz_points:
                #     if abs(xyz[0]) > 100. or abs(xyz[1]) > 100. or abs(xyz[2]) > 100. :
                #         print(xyz)

                # label = np.fromfile(str(os.path.join(self.labels_dir, labels_file_name)), dtype=np.float32)
                labels = pd.read_csv(str(os.path.join(self.labels_dir, labels_file_name)), \
                                    sep = " ", header=None)
                labels.columns = ["x", "y", "z", "dx", "dy", "dz", "yaw", "class"]                    
                print(str(labels_file_name), '\n')
                df_buf = pd.DataFrame(labels)
                print(df_buf)
                
                f = open(debug_file_abs, "r")
                print(f.read())

                line_set_array = []
                for x, y, z, dx, dy, dz, yaw, class_name in \
                    zip(df_buf["x"], df_buf["y"], df_buf["z"], df_buf["dx"], df_buf["dy"], df_buf["dz"], df_buf["yaw"], df_buf["class"]):
                    # print(x, y, z, dx, dy, dz, yaw, class_name)
                    corner_box = self.box_center_to_corner([x, y, z, dz, dy, dx, yaw])
                    # print(corner_box)

                    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
                    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                            [4, 5], [5, 6], [6, 7], [4, 7],
                            [0, 4], [1, 5], [2, 6], [3, 7]]

                    # Use the same color for all lines
                    colors = [[1, 0, 0] for _ in range(len(lines))]

                    line_set = open3d.geometry.LineSet()
                    line_set.points = open3d.utility.Vector3dVector(corner_box)
                    line_set.lines = open3d.utility.Vector2iVector(lines)
                    line_set.colors = open3d.utility.Vector3dVector(colors)
                    line_set_array.append(line_set)
                # # Create a visualization object and window
                # vis = open3d.visualization.Visualizer()
                # vis.create_window()

                # # Display the bounding boxes:
                # vis.add_geometry(corner_box)


                # print(xyz_points.shape, obj_points.shape)
                point_cloud.points = open3d.utility.Vector3dVector(xyz_points)
                frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0.0, 0.0, 0.0])
                open3d.visualization.draw_geometries([point_cloud, frame]+line_set_array)


        except KeyboardInterrupt:
            print('interrupted!')
            # open3d.draw_geometries([point_cloud])      
            # self.pcd_file_path_list.append(pcd_file_abs)
            # # print(pcd_file_abs)

            # base_name = os.path.splitext(pcd_file_name)[0]
            # labeled_json_file_abs = os.path.join(labeled_data_dir, base_name + '.json')
            # self.labeled_json_path_list.append(labeled_json_file_abs)

            # image_file_abs = os.path.join(camera_data_dir, base_name + '.jpg')
            # self.image_file_path_list.append(image_file_abs)

    def box_center_to_corner(self, box):
        # To return
        corner_boxes = np.zeros((8, 3))

        translation = box[0:3]
        h, w, l = box[3], box[4], box[5]
        rotation = box[6]

        # Create a bounding box outline
        bounding_box = np.array([
            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

        # Standard 3x3 rotation matrix around the Z axis
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0]])

        # Repeat the [x, y, z] eight times
        eight_points = np.tile(translation, (8, 1))

        # Translate the rotated bounding box by the
        # original center position to obtain the final box
        corner_box = np.dot(
            rotation_matrix, bounding_box) + eight_points.transpose()

        return corner_box.transpose()


if __name__ == '__main__':

    # current_path = os.path.abspath(__file__)
    openpcdet_root_path = "/OpenPCDet"
    # joined_root_path = os.path.join(openpcdet_root_path, "data","custom")

    # joined_root_path = os.path.join(openpcdet_root_path, "data","custom") #training
    # joined_root_path = os.path.join(openpcdet_root_path, "data","NIA", "custom_no_calib") #debug, no calib
    joined_root_path = os.path.join(openpcdet_root_path, "data","NIA", "custom") #debug, calib

    visualizer = NiaVisualizer(root_path=joined_root_path) 
    print("Finish")
    
