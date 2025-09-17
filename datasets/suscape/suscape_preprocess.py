import os
import json
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from suscape.dataset import SuscapeDataset
from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.visualization import dump_3d_bbox_on_image, color_mapper
import open3d as o3d
from pypcd import pypcd  # 尝试先用 pypcd 读 intensity，失败就 fallback

SUSCAPE_LABELS = [
    "Car",
    "Pedestrian",
    "Van",
    "Bus",
    "Truck",
    "ScooterRider",
    "Scooter",
    "BicycleRider",
    "Bicycle",
    "Motorcycle",
    "MotorcycleRider",
    "PoliceCar",
    "TourCar",
    "RoadWorker",
    "Child",
    "BabyCart",
    "Cart",
    "Cone",
    "FireHydrant",
    "SaftyTriangle",
    "PlatformCart",
    "ConstructionCart",
    "RoadBarrel",
    "TrafficBarrier",
    "LongVehicle",
    "BicycleGroup",
    "ConcreteTruck",
    "Tram",
    "Excavator",
    "Animal",
    "TrashCan",
    "ForkLift",
    "Trimotorcycle",
    "FreightTricycle",
    "Crane",
    "RoadRoller",
    "Bulldozer",
    "DontCare",
    "Misc",
    "Unknown",
    "Unknown1",
    "Unknown2",
    "Unknown3",
    "Unknown4",
    "Unknown5"
]


SUSCAP_NONRIGID_DYNAMIC_CLASSES = [
    "Pedestrian",
    "Child",
    "BicycleRider",
    "MotorcycleRider",
    "ScooterRider",
    "BicycleGroup",
    "BabyCart",
    "Cart"
]



SUSCAP_RIGID_DYNAMIC_CLASSES = [
    "Car",
    "Van",
    "Bus",
    "Truck",
    "Motorcycle",
    "PoliceCar",
    "TourCar",
    "ConcreteTruck",
    "Tram",
    "Excavator",
    "ForkLift",
    "Trimotorcycle",
    "FreightTricycle",
    "Crane",
    "RoadRoller",
    "Bulldozer",
    "LongVehicle"
]


SUSCAP_DYNAMIC_CLASSES = SUSCAP_NONRIGID_DYNAMIC_CLASSES + SUSCAP_RIGID_DYNAMIC_CLASSES

# valid_ring_cams = set([x.value for x in RingCameras])
# valid_stereo_cams = set([x.value for x in StereoCameras])

class SUScapeProcessor(object):

    """Process SUScape.
    
    LiDAR: 10Hz, Camera: 20Hz
    Since the LiDAR and Camera are not synchronized, we need to find the closest camera image for each LiDAR frame.
    Thus the actual frame rate of the processed data is 10Hz, which is aligned with the LiDAR.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
    """
    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys=[
            "camera",
            "lidar",
            "calib",
            "ego_pose",
            # "dynamic_masks",
            "label"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)
        
        # dataset = SuscapeDataset(load_dir)
        # print(process_id_list)
        # print(len(dataset.get_scene_names()), 'scenes')
        # print(dataset.get_scene_info("scene-000000"))
        # SUScape Provides 7 cameras, we process 5 of them
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "front",   # "xxx_0.jpg"
            "front_left",     # "xxx_1.jpg"
            "front_right",    # "xxx_2.jpg"
            "rear",      # "xxx_3.jpg"
            "rear_left",     # "xxx_4.jpg"
            "rear_right",      # "xxx_5.jpg"
        ]
        # cam_enums: List[Union[RingCameras, StereoCameras]] = []
        # for cam_name in self.cam_list:
        #     if cam_name in valid_ring_cams:
        #         cam_enums.append(RingCameras(cam_name))
        #     elif cam_name in valid_stereo_cams:
        #         cam_enums.append(StereoCameras(cam_name))
        #     else:
        #         raise ValueError("Must provide _valid_ camera names!")
        
        # Prepare dynamic objects' metadata
        self.load_dir = load_dir
        self.save_dir = f"{save_dir}"
        self.workers = int(workers)
        self.suscapeloader = SuscapeDataset(self.load_dir)
        # a list of tfrecord pathnames
        self.training_files = open("data/argoverse_train_list.txt").read().splitlines()
        self.log_pathnames = [
            f"{self.load_dir}/{f}" for f in self.training_files
        ]
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")
        
    def get_lidar_indices(self, log_id: str):
        sensor_cache = self.av2loader.sensor_cache
        lidar_indices = sensor_cache.xs(key="lidar", level=2).index
        lidar_indices_mask = lidar_indices.get_level_values('log_id') == log_id
        # get the positions of the lidar indices, get nonzeros
        lidar_indices = np.nonzero(lidar_indices_mask)[0]
        return lidar_indices
    
    def filter_lidar_indices(self, lidar_indices):
        """
        Filter lidar indices whose corresponding synchronized camera images are not complete.
        These usually happen at the beginning and the end of the sequence.
        """
        valid_list = []
        invalid_list = []
        for idx in lidar_indices:
            datum = self.av2loader[idx]
            sweep = datum.sweep
            
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            synchronized_imagery = datum.synchronized_imagery
            
            cnt = 0
            for _, cam in synchronized_imagery.items():
                if (
                    cam.timestamp_ns in timestamp_city_SE3_ego_dict
                    and sweep.timestamp_ns in timestamp_city_SE3_ego_dict
                ):
                    cnt += 1
                    
            if cnt != len(self.cam_list):
                invalid_list.append(idx)
                continue
            valid_list.append(idx)
        print(f"INFO: {len(invalid_list)} lidar indices filtered")
            
        return valid_list
        
    def convert_one(self, scene_idx):
        """Convert action for single file.

        Args:
            scene_idx (str): Scene index.
        """
        infos = self.suscapeloader.get_scene_info(scene_idx)

        frames = infos['frames']
        lidar_indices = self.get_lidar_indices(
            self.training_files[scene_idx]
        )
        lidar_indices = self.filter_lidar_indices(lidar_indices)        
        # process each frame
        num_frames = len(frames)
        for idx, frame_idx in tqdm(
            enumerate(frames), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True
        ):  

            datum = self.suscapeloader
            if "images" in self.process_keys:
                self.save_image(idx, scene_idx, frame_idx, infos)
            if "calib" in self.process_keys:
                self.save_calib(datum, scene_idx, frame_idx, infos)
            if "lidar" in self.process_keys:
                self.save_lidar(idx, scene_idx, frame_idx, infos)
            if "pose" in self.process_keys:
                self.save_pose(idx, scene_idx, frame_idx, infos)
            if "3dbox_vis" in self.process_keys:
                # visualize 3d box, debug usage
                self.visualize_3dbox(datum, scene_idx, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(idx, scene_idx, frame_idx, class_valid='all', infos=infos)
                self.save_dynamic_mask(idx, scene_idx, frame_idx, class_valid='human', infos=infos)
                self.save_dynamic_mask(idx, scene_idx, frame_idx, class_valid='vehicle', infos=infos)
                
        # sort and save objects info
        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(scene_idx)
            print(f"Processed instances info for {scene_idx}")
            
            # Save instances info and frame instances
            object_info_dir = f"{self.save_dir}/{str(scene_idx)[-3:]}/instances"
            with open(f"{object_info_dir}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{object_info_dir}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)
            
            # verbose: visualize the instances on the image (Debug Usage)
            if "objects_vis" in self.process_keys:
                self.visualize_dynamic_objects(
                    scene_idx, lidar_indices,
                    instances_info=instances_info,
                    frame_instances=frame_instances
                )
                print(f"Processed objects visualization for {scene_idx}")

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_image(self, idx, scene_idx, frame_idx, infos):
        """Parse and save the images in jpg format.

        Args:
            datum (:obj:`SuscapeDataset`): SUScape synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """

        for cam_name in infos['camera']:
            cam_idx = self.cam_list.index(cam_name)
            img_path = (
                f"{self.save_dir}/{str(scene_idx)[-3:]}/images/"
                + f"{idx:03d}_{str(cam_idx)}.jpg"
            )        
            src_path = self.load_dir+'/'+scene_idx+'/'+'camera'+'/'+cam_name+'/'+frame_idx+'.jpg'
            image = Image.open(src_path).convert("RGB")
            image.save(img_path)

    def save_calib(self, datum: SuscapeDataset, scene_idx, frame_idx, infos):
        """Parse and save the calibration data.

        Args:
            datum (:obj:`SuscapeDataset`): SUScape synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        # synchronized_imagery = datum.synchronized_imagery
        for cam_name in infos['camera']:
            idx = self.cam_list.index(cam_name)
            src_path = self.load_dir+'/'+scene_idx+'/calib'+'/camera'+'/'+cam_name+'.json'
            # 读取 JSON
            with open(src_path, "r") as f:
                calib = json.load(f)
            # lidar_to_camera 转成 4×4 矩阵
            c2v = np.array(calib["lidar_to_camera"]).reshape(4, 4)

            # intrinsic 转成 3×3 矩阵
            K = np.array(calib["intrinsic"]).reshape(3, 3) 
            # 按你需要的格式展开
            intrinsics = np.array([
                K[0, 0],  # fx
                K[1, 1],  # fy
                K[0, 2],  # cx
                K[1, 2],  # cy
                0.0, 0.0, 0.0, 0.0, 0.0
            ])   
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx)[-3:]}/extrinsics/"
                + f"{str(idx)}.txt",
                c2v,
            )
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx)[-3:]}/intrinsics/"
                + f"{str(idx)}.txt",
                intrinsics,
            )
    def convert_pcd_to_bin(self, src_path):
        try:
            # 先用 pypcd 读取，能取出 intensity 字段
            pc = pypcd.PointCloud.from_path(src_path)
            xyz = np.vstack((pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z'])).T
            if 'intensity' in pc.pc_data.dtype.names:
                intensity = pc.pc_data['intensity'].reshape(-1, 1)
            else:
                intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
            point_cloud = np.column_stack((xyz, intensity))
        except Exception:
            # 如果 pypcd 失败，就用 open3d 读，只保留坐标
            pcd = o3d.io.read_point_cloud(src_path)
            xyz = np.asarray(pcd.points)
            intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
            point_cloud = np.column_stack((xyz, intensity))
        
        return point_cloud.astype(np.float32)
    def save_lidar(self, idx, scene_idx, frame_idx, infos):
        """Parse and save the lidar data in psd format.

        Args:
            datum (:obj:`SuscapeDataset`): SUScape synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        src_path = self.load_dir+'/'+scene_idx+'/'+'lidar'+'/'+frame_idx+'.pcd'
        
        point_cloud = self.convert_pcd_to_bin(src_path)
        
        pc_path = (
            f"{self.save_dir}/"
            + f"{str(scene_idx)[-3:]}/lidar/{idx:03d}.bin"
        )
        # 创建父目录
        os.makedirs(os.path.dirname(pc_path), exist_ok=True)
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_pose(self, datum: SuscapeDataset, scene_idx, frame_idx, infos):
        """Parse and save the pose data.

        Args:
            datum (:obj:`SuscapeDataset`): SUScape synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        src_path = self.load_dir+'/'+scene_idx+'/'+'lidar_pose'+'/'+frame_idx+'.json'
        # 读取 JSON
        with open(src_path, "r") as f:
            data = json.load(f)
        lidar_pose = np.array(data["lidarPose"], dtype=np.float64).reshape(4, 4)
        np.savetxt(
            f"{self.save_dir}/{str(scene_idx)[-3:]}/ego_pose/"
            + f"{idx:03d}.txt",
            lidar_pose,
        )
        
    def visualize_3dbox(self, datum: SuscapeDataset, scene_idx, frame_idx):
        """DEBUG: Visualize the 3D bounding box on the image.
        Visualize the 3D bounding box all with the same COLOR.
        If you want to visualize the 3D bounding box with different colors, please use the `visualize_dynamic_objects` function.

        Args:
            datum (:obj:`SuscapeDataset`): SUScape synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        annotations = datum.annotations
        synchronized_imagery = datum.synchronized_imagery
        for idx, cam_name in enumerate(self.cam_list):
            cam = synchronized_imagery[cam_name]
            if annotations is not None:
                img_plotted = annotations.project_to_cam(
                    img=cam.img,
                    cam_model=cam.camera_model,
                )
            else:
                img_plotted = cam.img.copy()
            
            # save
            img_path = (
                f"{self.save_dir}/{str(scene_idx)[-3:]}/3dbox_vis/"
                + f"{str(frame_idx)[-3:]}_{str(idx)}.jpg"
            )
            Image.fromarray(
                img_plotted[:, :, [2, 1, 0]]
            ).save(img_path)
            
    def visualize_dynamic_objects(
        self, scene_idx, lidar_indices,
        instances_info, frame_instances
    ):
        """DEBUG: Visualize the dynamic objects'box with different colors on the image.

        Args:
            scene_idx (str): Current file index.
            lidar_indices (list): List of lidar indices.
            instances_info (dict): Instances information.
            frame_instances (dict): Frame instances.
        """
        output_path = f"{self.save_dir}/{str(scene_idx)[-3:]}/instances/debug_vis"
        
        print("Visualizing dynamic objects ...")
        for frame_idx, lidar_idx in tqdm(
            enumerate(lidar_indices), desc=f"Visualizing dynamic objects of scene {scene_idx} ...", total=len(lidar_indices), dynamic_ncols=True
        ):
            datum = self.av2loader[lidar_idx]
            synchronized_imagery = datum.synchronized_imagery
            for cam_name, cam in synchronized_imagery.items():
                cam_idx = self.cam_list.index(cam_name)
                img_path = (
                    f"{self.save_dir}/{str(scene_idx)[-3:]}/images/"
                    + f"{str(frame_idx)[-3:]}_{str(cam_idx)}.jpg"
                )
                canvas = np.array(Image.open(img_path))
                
                if frame_idx in frame_instances:
                    objects = frame_instances[frame_idx]
                    
                    if len(objects) == 0:
                        img_plotted = canvas
                    else:
                        lstProj2d = []
                        color_list = []
                        for obj_id in objects:
                            idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
                            o2w = np.array(
                                instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
                            )
                            length, width, height = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
                            half_dim_x, half_dim_y, half_dim_z = length/2.0, width/2.0, height/2.0
                            corners = np.array(
                                [[half_dim_x, half_dim_y, -half_dim_z],
                                [half_dim_x, -half_dim_y, -half_dim_z],
                                [-half_dim_x, -half_dim_y, -half_dim_z],
                                [-half_dim_x, half_dim_y, -half_dim_z],
                                [half_dim_x, half_dim_y, half_dim_z],
                                [half_dim_x, -half_dim_y, half_dim_z],
                                [-half_dim_x, -half_dim_y, half_dim_z],
                                [-half_dim_x, half_dim_y, half_dim_z]]
                            )
                            corners = (o2w[:3, :3] @ corners.T + o2w[:3, [3]]).T
                            v2w = datum.timestamp_city_SE3_ego_dict[cam.timestamp_ns]
                            w2v = v2w.inverse()
                            corners_in_ego = w2v.transform_point_cloud(corners)
                            
                            projected_points2d, _, ok = cam.camera_model.project_ego_to_img(
                                corners_in_ego # cuboid corners in ego frame
                            )
                            projected_points2d = projected_points2d.tolist()
                            if all(ok):
                                lstProj2d.append(projected_points2d)
                                color_list.append(color_mapper(obj_id))
                                
                        lstProj2d = np.asarray(lstProj2d)
                        img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=color_list)
                
                img_path = (
                    f"{output_path}/"
                    + f"{str(frame_idx)[-3:]}_{str(cam_idx)}.jpg"
                )
                Image.fromarray(img_plotted).save(img_path)

    def save_dynamic_mask(self, idx, scene_idx, frame_idx, class_valid='all', infos=None):
        # print(infos.keys())
        """Parse and save the segmentation data.

        Args:
            datum (:obj:`SuscapeDataset`): SUScape synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
            class_valid (str): Class valid for dynamic mask.
        """
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = SUSCAP_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = SUSCAP_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = SUSCAP_RIGID_DYNAMIC_CLASSES
        mask_dir = f"{self.save_dir}/{str(scene_idx)[-3:]}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        for cam_name in infos['camera']:
            cam_idx = self.cam_list.index(cam_name)
            dynamic_mask_path = os.path.join(mask_dir, f"{idx:03d}_{str(cam_idx)}.png") 

            src_path = self.load_dir+'/'+scene_idx+'/'+'label_fusion/camera'+'/'+cam_name+'/'+frame_idx+'.json'
            # 读取 JSON
            with open(src_path, "r") as f:
                data = json.load(f)
            H, W = 1536, 2048
            dynamic_mask = np.zeros((H, W), dtype=np.uint8)
            objs = data.get("objs", [])
            for obj in objs:
                category = obj.get("obj_type", "")
                if category not in  VALID_CLASSES:
                    continue
                rect = obj.get("rect", {})
                x1, y1 = int(rect.get("x1", 0)), int(rect.get("y1", 0))
                x2, y2 = int(rect.get("x2", 0)), int(rect.get("y2", 0))
                # 防止越界
                x1 = np.clip(x1, 0, W - 1)
                x2 = np.clip(x2, 0, W - 1)
                y1 = np.clip(y1, 0, H - 1)
                y2 = np.clip(y2, 0, H - 1)
              
              
                if x2 <= x1 or y2 <= y1:
                    continue


                # 填充 mask
                dynamic_mask[y1:y2, x1:x2] = 255

            # 保存 mask
            Image.fromarray(dynamic_mask, mode="L").save(dynamic_mask_path)
            
    def save_objects(self, lidar_indices: List[int]):
        """Parse and save the objects annotation data.
        
        Args:
            lidar_indices (list): List of lidar indices.
        """
        instances_info, frame_instances = {}, {}
        for frame_idx, lidar_idx in enumerate(lidar_indices):
            datum = self.av2loader[lidar_idx]
            annotations = datum.annotations
            sweep = datum.sweep
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            
            frame_instances[frame_idx] = []
            for cuboid_idx in range(len(annotations)):
                cuboid = annotations[cuboid_idx]
                track_id, label = cuboid.track_uuid, cuboid.category
                if label not in SUSCAP_DYNAMIC_CLASSES:
                    continue
                
                if track_id not in instances_info:
                    instances_info[track_id] = dict(
                        id=track_id,
                        class_name=label,
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        }
                    )
                
                o2v = cuboid.dst_SE3_object.transform_matrix
                v2w = timestamp_city_SE3_ego_dict[sweep.timestamp_ns].transform_matrix
                # [object to  world] transformation matrix
                o2w = v2w @ o2v
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [cuboid.length_m, cuboid.width_m, cuboid.height_m]
                
                instances_info[track_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[track_id]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                instances_info[track_id]['frame_annotations']['box_size'].append(dimension)
                
                frame_instances[frame_idx].append(track_id)

        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        # Update keys in frame_instances
        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]

        return new_instances_info, new_frame_instances

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/sky_masks", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/intrinsics", exist_ok=True)
            if "pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/lidar", exist_ok=True)
            if "3dbox_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/3dbox_vis", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/dynamic_masks", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/instances", exist_ok=True)
            if "objects_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i)[-3:]}/instances/debug_vis", exist_ok=True)
