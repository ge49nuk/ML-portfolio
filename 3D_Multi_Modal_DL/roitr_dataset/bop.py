import os
import numpy as np
from torch.utils.data import Dataset
import torch
import json
from roitr_dataset.common import normal_redirect
from roitr_lib.utils import to_o3d_pcd, depth_to_world, rle_decode, pps_sampling, distance_to_nearest_neighbor, to_ply_file, calc_pc_radius, remove_outliers
import open3d as o3d
import cv2
import copy
from scipy.spatial.transform import Rotation


class BOP(Dataset):
    '''
    Load subsampled coordinates, relative rotation and translation
    Output (torch.Tensor):
    src_pcd: (N, 3) source point cloud
    tgt_pcd: (M, 3) target point cloud
    src_node_xyz: (n, 3) nodes sparsely sampled from source point cloud
    tgt_node_xyz: (m, 3) nodes sparsely sampled from target point cloud
    rot: (3, 3)
    trans: (3, 1)
    correspondences: (?, 3)
    '''

    def __init__(self, config, split):      
        self.base_dir = config.root
        if split == "test":
            self.data_path = os.path.join(self.base_dir, 'test')
            if not os.path.exists(self.data_path):
                self.data_path = os.path.join(self.base_dir, 'train_pbr')
            if "tless" in self.base_dir:
                self.data_path = os.path.join(self.base_dir, 'test_primesense')
        else:
            self.data_path = os.path.join(self.base_dir, 'train_pbr')
            if not os.path.exists(self.data_path):
                self.data_path = os.path.join(self.base_dir, 'test')
        self.config = config
        self.split = split
        self.view_point = np.array([0., 0., 0.])
        self.points_lim = 8192
        self.data_augmentation = config.data_augmentation
        self.rot_factor = 1.
        self.av_visib = []
        
        self.init_dataset(split)
                
    def init_dataset(self, split):
        print(f"Loading {split} data...")
        self.data = []
        used_rgb_ids = set()
        used_obj_ids = set()
        
        bop_scenes = os.listdir(self.data_path)
        train_end = int(self.config.train_val_split * len(bop_scenes))
        if split == "train":
            bop_scenes = bop_scenes[:train_end]
        elif split == "val":
            bop_scenes = bop_scenes[train_end:]
            
        for folder in bop_scenes[:]: # BOP scenes
            with open(os.path.join(self.data_path, folder, "scene_gt.json"), "r") as f:
                scene_gt = json.load(f)
            with open(os.path.join(self.data_path, folder, "scene_gt_info.json"), "r") as f:
                scene_gt_info = json.load(f)
            with open(os.path.join(self.data_path, folder, "scene_camera.json"), "r") as f:
                scene_camera = json.load(f)
            keys = list(scene_gt.keys())

            # Iterate through images
            for key in keys[:]: 
                gt_row = scene_gt[key]
                if key not in scene_gt_info:
                    continue
                gt_info_row = scene_gt_info[key]
                camera_row = scene_camera[key]

                # Iterate through queried objects
                for i, object in enumerate(gt_row[:]):  
                    gt_info_object = gt_info_row[i]
                    if gt_info_object["visib_fract"] < self.config.min_visib_fract:
                        continue
                    self.av_visib.append(gt_info_object["visib_fract"])
                    # initialize dataset sample
                    data_dict = {"rgb_id": int(folder)*10000 + int(key)}
                    data_dict["translation"] = np.array(object["cam_t_m2c"]).reshape(3,1)
                    data_dict["rotation"] = np.array(object["cam_R_m2c"]).reshape(3,3)
                    data_dict["cam_K"] = np.array(camera_row["cam_K"]).reshape(3,3)
                    data_dict["depth_scale"] = camera_row["depth_scale"]
                    bbox = np.array(gt_info_object["bbox_visib"], dtype=np.float32)
                    if bbox[2] * bbox[3] < 4096: # Bbox is too small
                        continue
                    data_dict["bbox"] = bbox.astype(int)
                    data_dict["obj_id"] = object["obj_id"]
                    data_dict["og_id"] = object["obj_id"]
                    mask_file = key.zfill(6) + "_" + str(i).zfill(6) + ".png"
                    data_dict["mask_path"] = os.path.join(self.data_path, folder, "mask_visib", mask_file)
                    
                    # add distractor object
                    if self.config.mode == "test":
                        if object["obj_id"] <= self.config.num_test_objs:
                            self.data.append(data_dict) 
                            if self.config.add_distractor: 
                                d_id = object["obj_id"] % self.config.num_test_objs + 1
                                d_copy = copy.deepcopy(data_dict)
                                d_copy["obj_id"] = d_id
                                self.data.append(d_copy) 
                    else:
                        self.data.append(data_dict) 
                        
                    used_rgb_ids.add(data_dict["rgb_id"])
                    used_obj_ids.add(data_dict["og_id"])
        
        print(f"{split} dataset: {len(self.data)} samples, {len(used_obj_ids)} meshes, {len(used_rgb_ids)} images, Average visib_fract: {np.mean(self.av_visib)}")           
    
    def load_depth_img(self, rgb_id):
        folder =  str(rgb_id//10000).zfill(6)
        PATH = os.path.join(self.data_path, folder, "depth")
        fn = str(rgb_id%10000).zfill(6) + ".png"
        image = cv2.imread(os.path.join(PATH, fn), cv2.IMREAD_ANYDEPTH).T   
        return image
    
    def load_mesh(self, obj_id):
        file = "obj_" + str(obj_id).zfill(6) + ".ply"
        mesh = o3d.io.read_triangle_mesh(os.path.join(self.base_dir, "models", file))                
        pcd = mesh.sample_points_uniformly(number_of_points=self.points_lim)
        vertices = np.asarray(pcd.points)
        return vertices        

    def get_binary_mask(self, data_dict):
        mask = cv2.imread(data_dict["mask_path"], cv2.IMREAD_ANYDEPTH).T / 255.
        return mask
            
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        depth = self.load_depth_img(data_dict["rgb_id"])
        rot = np.linalg.inv(data_dict["rotation"])
        trans = -data_dict["translation"]
        
        ##################################################################################################
        # Apply segmentation mask
        mask = self.get_binary_mask(data_dict)
        src_pcd = depth_to_world(depth, data_dict["cam_K"], data_dict["depth_scale"], data_dict["bbox"], mask, True)#self.split == "train" and self.data_augmentation) 
        tgt_pcd = self.load_mesh(data_dict["obj_id"])

        ##################################################################################################
        # Pre-process point clouds and set matching radius

        # normalize point clouds
        center_tgt = np.mean(tgt_pcd, axis=0)
        tgt_pcd -= center_tgt
        trans -= rot.T @ center_tgt.reshape(3,1)
        rad_tgt = calc_pc_radius(tgt_pcd)
        tgt_pcd /= rad_tgt

        src_pcd, center_src, _ = remove_outliers(src_pcd, rad_tgt)
        src_pcd -= center_src
        trans += center_src.reshape(3,1)
        src_pcd /= rad_tgt
        trans /= rad_tgt

        if src_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(src_pcd.shape[0])[:self.points_lim]
            src_pcd = src_pcd[idx]
        
        ##################################################################################################
        # Data augmentation
        if self.data_augmentation and self.split == "train":
            # simulate depth sensor inaccuracy
            accuracy = 2. / rad_tgt # sensor has 2mm accuracy
            src_pcd[:, 2] = np.round(src_pcd[:, 2] / accuracy) * accuracy

            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                tgt_pcd = np.matmul(tgt_pcd, rot_ab)

                rot = np.matmul(rot_ab.T, rot)
            else:
                src_pcd = np.matmul(src_pcd, rot_ab)

                rot = np.matmul(rot, rot_ab)
                trans = np.matmul(rot_ab.T, trans)
            # src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * (matching_radius / 2)

        # gt_src = np.matmul(src_pcd + trans.T, rot.T)
        # to_ply_file(gt_src, tgt_pcd)
        # exit(0)  

        ##################################################################################################
        # Normal estimation
        o3d_src_pcd = to_o3d_pcd(src_pcd)
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals)
        src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
        o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        tgt_normals = np.asarray(o3d_tgt_pcd.normals)
        tgt_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=self.view_point)
        src_feats = np.ones(shape=(src_pcd.shape[0], 1))
        tgt_feats = np.ones(shape=(tgt_pcd.shape[0], 1))

        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_normals.astype(np.float32), tgt_normals.astype(np.float32),\
               src_feats.astype(np.float32), tgt_feats.astype(np.float32),\
               rot.astype(np.float32), trans.astype(np.float32),\
               src_pcd.astype(np.float32), None, data_dict["obj_id"], data_dict["og_id"], mask, data_dict["bbox"]

    def __len__(self):
        return len(self.data)