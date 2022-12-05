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
from pypcd import pypcd
from cam2lidar import reverseXcYcZc

class NIA2KittiStyle():
    def __init__(self, root_path = None):
        self.root = root_path
        self.data_3D_dir = os.path.join(self.root, '3D')
        self.scenario_dir = os.listdir(self.data_3D_dir) #sorted(glob.glob(self.root))
        
        if not os.path.exists(os.path.join(self.root, "custom" ,'points')):
            os.makedirs(os.path.join(self.root, "custom" ,'points'))
        if not os.path.exists(os.path.join(self.root, "custom" ,'labels')):
            os.makedirs(os.path.join(self.root, "custom" ,'labels'))
        if not os.path.exists(os.path.join(self.root, "custom" ,'cameras')):
            os.makedirs(os.path.join(self.root, "custom" ,'cameras'))
        if not os.path.exists(os.path.join(self.root, "custom" ,'ImageSets')):
            os.makedirs(os.path.join(self.root, "custom" ,'ImageSets'))
        if not os.path.exists(os.path.join(self.root, "custom" ,'debug')):
            os.makedirs(os.path.join(self.root, "custom" ,'debug'))
        if not os.path.exists(os.path.join(self.root, "custom" ,'calib')):
            os.makedirs(os.path.join(self.root, "custom" ,'calib'))

        print(self.root)
        print(self.scenario_dir)

        #scenarios : day_city, night_city, day_highway, night_highway, bad_weather
        self.scenario = []
        self.scenario_date_time = []
        self.date_time = []
        self.pcd_file_path_list = []
        self.labeled_json_path_list = []
        self.image_file_path_list = []
        self.calib_json_path_list = []
        for scenario in self.scenario_dir:
            self.scenario.append(scenario)
            # print(scenario)
            
            current_scenario_dir = os.path.join(self.data_3D_dir, scenario)
            for date_time_dir in os.listdir(current_scenario_dir):
                self.scenario_date_time.append(os.path.join(scenario, date_time_dir))
                self.date_time.append(date_time_dir)
                # print(os.path.join(scenario, date_time_dir))

        self.FileFinder()
        self.FileConverter()

    def FileFinder(self):        
        for scenario_date_time, date_time in zip(self.scenario_date_time, self.date_time):
            original_data_dir = os.path.join(self.root, 'original_data', scenario_date_time, 'sensor_raw_data', 'lidar')
            camera_data_dir = os.path.join(self.root, 'original_data', scenario_date_time, 'sensor_raw_data', 'camera')
            calib_data_dir = os.path.join(self.root, 'original_data', scenario_date_time)
            labeled_data_dir = os.path.join(self.root, '3D', scenario_date_time, 'sensor_raw_data', 'camera')            
            # print(original_data_dir)
            # print("scenario_date_time : " , scenario_date_time)
            # print("date_time : " , date_time)

            for pcd_file_name in os.listdir(original_data_dir):
                pcd_file_abs = os.path.join(original_data_dir, pcd_file_name)                
                self.pcd_file_path_list.append(pcd_file_abs)
                # print(pcd_file_abs)

                base_name = os.path.splitext(pcd_file_name)[0]
                labeled_json_file_abs = os.path.join(labeled_data_dir, base_name + '.json')
                self.labeled_json_path_list.append(labeled_json_file_abs)

                image_file_abs = os.path.join(camera_data_dir, base_name + '.jpg')
                self.image_file_path_list.append(image_file_abs)
        
                calib_file_abs = os.path.join(calib_data_dir, date_time + '_meta_data.json')
                self.calib_json_path_list.append(calib_file_abs)

    def FileConverter(self):
        # print(self.pcd_file_path_list)
        # print(self.labeled_json_path_list)
        i = 0
        from tqdm import tqdm
        for pcd_file, labeled_json, image_file, calib_file in tqdm(zip(self.pcd_file_path_list, self.labeled_json_path_list, self.image_file_path_list, self.calib_json_path_list)):
            print('\n')
            data = self.jsonOpen(labeled_json)
            annotation_list = data['annotations']
            if len(annotation_list) != 0:
                # check dimension is null
                is_dimension_null = True
                for annotation in annotation_list:
                    if annotation["dimension"] is not None:
                        is_dimension_null = False

                if is_dimension_null is True:
                    continue
                
                file_name = "{0:06d}".format(i)
                file_path = os.path.join(self.root, "custom" ,'labels', file_name + '.txt')
                file_buf = open(file_path, 'w')

                # converted path & origin path
                debug_path = os.path.join(self.root, "custom" ,'debug', file_name + '.txt')
                debug_buf = open(debug_path, 'w')
                debug_content = file_name + '.npy' + '\t' + pcd_file
                debug_buf.write(debug_content)

                print("pcd path: " , pcd_file)
                print("converted path : ", file_path)

                # image copy for debug
                src_image = image_file
                dst_image = os.path.join(self.root, "custom" ,'cameras', file_name + '.jpg')
                shutil.copyfile(src_image, dst_image)

                # calib 
                src_calib = calib_file
                dst_calib = os.path.join(self.root, "custom" ,'calib', file_name + '.json')
                shutil.copyfile(src_calib, dst_calib)

                calib_data = self.jsonOpen(calib_file)
                # front_lidar_calib_info = calib_data["calibration"]["lidar"]["front"]
                front_lidar_calib_info = calib_data["calibration"]["camera"]["front"]["Extrinsic"]
                # print(front_lidar_calib_info)
                calib_mat = reverseXcYcZc(front_lidar_calib_info, "camera")
                # print(calib_mat)

                # const result_4dim = [result[0], result[1], result[2], [1]];
                # const xyz1 = math.multiply(math.inv(total_value), result_4dim);
                # // const result = this.multiplyMatrices(total_value, [[X], [Y], [Z], [1]]);
                # // 카메라 회전 값

                # Point-cloud data -> pcd to npy
                point_cloud = pypcd.PointCloud.from_path(pcd_file)
                pc_data = point_cloud.pc_data
                pc_array = np.column_stack([pc_data["x"], pc_data["y"], pc_data["z"], pc_data["intensity"]])
                pc_array = pc_array.reshape(-1,4)
                pc_array = pc_array.astype('float32')                
                np.save(os.path.join(self.root, "custom" ,'points', file_name + '.npy'), pc_array)
                # print(pc_array)
                
                # Labeling data
                for annotation in annotation_list:
                    if annotation["location"] is not None :
                        # format: [x y z dx dy dz heading_angle category_name]
                        # content = "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %s\n" % \
                        #     (annotation["location"][0], annotation["location"][1], annotation["location"][2], \
                        #     annotation["dimension"][2], annotation["dimension"][0], annotation["dimension"][1], \
                        #     annotation["yaw"], annotation["class"])

                        # Calibration
                        result_4dim = np.array([[annotation["location"][0]], [annotation["location"][1]], [annotation["location"][2]], [1.0]])
                        # print(result_4dim)
                        xyz1 = np.matmul(np.linalg.inv(calib_mat), result_4dim)
                        # print(xyz1[0][0], xyz1[1][0], xyz1[2][0])
                        content = "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %s\n" % \
                            (xyz1[0][0], xyz1[1][0], xyz1[2][0], \
                            annotation["dimension"][2], annotation["dimension"][0], annotation["dimension"][1], \
                            annotation["yaw"], annotation["class"])

                        file_buf.write(content)               

                file_buf.close()
                debug_buf.close()
                i = i + 1

        max_num = i

        training_set_num = max_num * 0.8
        trainset_file_path = os.path.join(self.root, "custom" ,'ImageSets', "train" + '.txt')
        valset_file_path = os.path.join(self.root, "custom" ,'ImageSets', "val" + '.txt')

        training_file_buf = open(trainset_file_path, 'w')
        val_file_buf = open(valset_file_path, 'w')
        for num in range(max_num):
            if num < training_set_num:
                content_buf = "{0:06d}\n".format(num)
                training_file_buf.write(content_buf)
                # print(num)
            else : 
                content_buf = "{0:06d}\n".format(num)
                val_file_buf.write(content_buf)

        training_file_buf.close()        
        val_file_buf.close()


    def jsonOpen(self, path): 
        try:
            with open(path, 'r', encoding='UTF8') as file:
                data = json.load(file)
        except:
            with open(path, 'r', encoding='CP949') as file:
                data = json.load(file)
        return (data)


        # dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # create_custom_infos(
        #     dataset_cfg=dataset_cfg,
        #     class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
        #     data_path=ROOT_DIR / 'data' / 'custom',
        #     save_path=ROOT_DIR / 'data' / 'custom',
        # )


if __name__ == '__main__':

    # current_path = os.path.abspath(__file__)
    openpcdet_root_path = "/OpenPCDet"
    joined_root_path = os.path.join(openpcdet_root_path, "data","NIA")

    converter = NIA2KittiStyle(root_path=joined_root_path) 
    print("Finish")
    
