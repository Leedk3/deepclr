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

        print(self.root)
        print(self.scenario_dir)

        #scenarios : day_city, night_city, day_highway, night_highway, bad_weather
        self.scenario = []
        self.scenario_date_time = []
        self.pcd_file_path_list = []
        self.labeled_json_path_list = []
        self.image_file_path_list = []
        for scenario in self.scenario_dir:
            self.scenario.append(scenario)
            # print(scenario)
            
            current_scenario_dir = os.path.join(self.data_3D_dir, scenario)
            for date_time_dir in os.listdir(current_scenario_dir):
                self.scenario_date_time.append(os.path.join(scenario, date_time_dir))
                # print(os.path.join(scenario, date_time_dir))

        self.FileFinder()
        self.FileConverter()

    def FileFinder(self):        
        for scenario_date_time in self.scenario_date_time:
            original_data_dir = os.path.join(self.root, 'original_data', scenario_date_time, 'sensor_raw_data', 'lidar')
            camera_data_dir = os.path.join(self.root, 'original_data', scenario_date_time, 'sensor_raw_data', 'camera')
            labeled_data_dir = os.path.join(self.root, '3D', scenario_date_time, 'sensor_raw_data', 'camera')            
            # print(original_data_dir)
            for pcd_file_name in os.listdir(original_data_dir):
                pcd_file_abs = os.path.join(original_data_dir, pcd_file_name)                
                self.pcd_file_path_list.append(pcd_file_abs)
                # print(pcd_file_abs)

                base_name = os.path.splitext(pcd_file_name)[0]
                labeled_json_file_abs = os.path.join(labeled_data_dir, base_name + '.json')
                self.labeled_json_path_list.append(labeled_json_file_abs)

                image_file_abs = os.path.join(camera_data_dir, base_name + '.jpg')
                self.image_file_path_list.append(image_file_abs)

    def FileConverter(self):
        # print(self.pcd_file_path_list)
        # print(self.labeled_json_path_list)
        i = 0
        from tqdm import tqdm
        for pcd_file, labeled_json, image_file in tqdm(zip(self.pcd_file_path_list, self.labeled_json_path_list, self.image_file_path_list)):
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
                
                save_it = False
                print(pcd_file)
                file_name = "{0:06d}".format(i)
                file_path = os.path.join(self.root, "custom" ,'labels', file_name + '.txt')
                file_buf = open(file_path, 'w')
                debug_path = os.path.join(self.root, "custom" ,'debug', file_name + '.txt')
                debug_buf = open(debug_path, 'w')
                print(file_path)

                src_image = image_file
                dst_image = os.path.join(self.root, "custom" ,'cameras', file_name + '.jpg')
                shutil.copyfile(src_image, dst_image)
                point_cloud = pypcd.PointCloud.from_path(pcd_file)
                pc_data = point_cloud.pc_data
                pc_array = np.column_stack([pc_data["x"], pc_data["y"], pc_data["z"], pc_data["intensity"]])

                pc_array = pc_array.reshape(-1,4)
                pc_array = pc_array.astype('float32')
                
                np.save(os.path.join(self.root, "custom" ,'points', file_name + '.npy'), pc_array)
                # print(pc_array)

                debug_content = file_name + '.npy' + '\t' + pcd_file
                debug_buf.write(debug_content)

                for annotation in annotation_list:
                    if annotation["location"] is not None :
                        # KITTI Format
                        # class names, truncation, occlusion, alpha, bbox, 3-D dim, Location, rot_y
                        #PCD to NPY
                        # point_cloud = open3d.io.read_point_cloud(pcd_file)
                        # pc_array = np.asarray(point_cloud.points)
                        # print('=========')
                        # print(i, pcd_file, labeled_json)
                        # print("%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" 
                        #       % (annotation["class"], annotation["attribute"]["truncated"], annotation["attribute"]["occluded"], \
                        #       annotation["yaw"], annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3], \
                        #       annotation["dimension"][0], annotation["dimension"][1], annotation["dimension"][2], \
                        #       annotation["location"][0], annotation["location"][1], annotation["location"][2], 0))
                        # content = "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (annotation["class"], annotation["attribute"]["truncated"], annotation["attribute"]["occluded"], \
                        #       annotation["yaw"], annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3], \
                        #       annotation["dimension"][0], annotation["dimension"][1], annotation["dimension"][2], \
                        #       annotation["location"][0], annotation["location"][1], annotation["location"][2], 0)

                        # format: [x y z dx dy dz heading_angle category_name]
                        content = "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %s\n"% \
                            (annotation["location"][0], annotation["location"][1], annotation["location"][2], \
                            annotation["dimension"][2], annotation["dimension"][0], annotation["dimension"][1], \
                            annotation["yaw"], annotation["class"])
                        file_buf.write(content)

                        # else :
                        #     # print("%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" 
                        #     #       % ("DontCare", -1, -1, \
                        #     #       -1, annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3], \
                        #     #       -1, -1, -1, \
                        #     #       -1000, -1000, -1000, 0))
                        #     # content = "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ("DontCare", -1, -1, \
                        #     #       -1, annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3], \
                        #     #       -1, -1, -1, \
                        #     #       -1000, -1000, -1000, 0)
                        #     # format: [x y z dx dy dz heading_angle category_name]
                        #     content = "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %s" % \
                        #          (-1000, -1000, -1000, \
                        #           -1, -1, -1, \
                        #           -1, "DontCare")
                        #     file_buf.write(content)                    

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

        # if(os.path.exists(os.path.join(self.root, '3D'))):
        #     print("3D exist")

        # if(os.path.exists(os.path.join(self.root, '2D'))):
        #     print("2D exist")

        # if(os.path.exists(os.path.join(self.root, '2D'))):
        #     print("2D exist")





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
    
