""" Predicting grasp pose from the input of point_cloud.
    Author: Hongrui-Zhu
    Note: 
        Pay attention to modifying camera parameters("self.camera_width"  "self.camera_high" "self.intrinsic" "self.factor_depth") to adapt to hardware
"""

import copy
import os
from typing import Optional, Tuple
import numpy as np
import open3d as o3d

import torch
import yaml
from graspnetAPI import GraspGroup

from graspnet.models.graspnet import GraspNet as graspnet, pred_decode
from graspnet.utils.collision_detector import ModelFreeCollisionDetector
from graspnet.utils.quaternion_utils import rota_to_quat
from scipy.spatial.transform import Rotation
def pose2mat(pose: Tuple[np.ndarray, np.ndarray]):
    translation, quaternion = pose
    # build rotation matrix
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    # build trans matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

from dataclasses import dataclass

# class GraspPredictor:
#     #########   Factory method   ############
#     _registry = {}

#     def __init_subclass__(cls, model, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls._registry[model] = cls

#     def __new__(cls, model: str, **kwargs):
#         subclass = cls._registry[model]
#         obj = object.__new__(subclass)
#         return obj

#     ######### Factory method end ############

#     def __init__(self, model, **kwargs) -> None:
#         """Create GraspModel with specific model

#         Args:
#             model (str): type of model
#         """
#         self.model: str = model
#         config_path = os.environ.get('LM_CONFIG')
#         with open(config_path, 'r') as f:
#             local_config = yaml.load(f, Loader=yaml.FullLoader)
#         extra = local_config.get(model, {})
#         self.kwargs = {**kwargs, **extra}
#         self.ckpt_dir = os.environ.get('CKPT_DIR', '/opt/ckpts')

#     def infer(self, point_cloud: np.ndarray, workspace_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

#         raise NotImplementedError
#         return (translation, quaternion)  # translation[3], quaternion[4]


class GraspNet():

    def __init__(self, model, **kwargs) -> None:
        """
        Initialize the model and load checkpoints (if needed)
        """
        # super().__init__(model, **kwargs)
        # load model
        config_path = os.environ.get('LM_CONFIG')
        with open(config_path, 'r') as f:
            local_config = yaml.load(f, Loader=yaml.FullLoader)
        extra = local_config.get(model, {})
        self.kwargs = {**kwargs, **extra}
        self.ckpt_dir = os.environ.get('CKPT_DIR', '/opt/ckpts')
        self.checkpoint_path = os.path.join(self.ckpt_dir, self.kwargs['ckpt'])
        self.net = self.get_net()
        # net params
        self.num_point = 20000
        self.num_view = 300
        self.collision_thresh = 0.01
        self.voxel_size = 0.01
        self.net = self.get_net()
        print("GraspNet initialized")

    # prompt: Optional[str] = None
    def infer(self, point_cloud: np.ndarray, workspace_mask) -> Tuple[np.ndarray, np.ndarray]:
        """Obtain the target grasp pose given a point cloud
        Args:
            point_cloud: the point cloud array with shape (N, 3) and dtype np.float64
                Each of the N entry is the cartesian coordinates in the world base
            workspace_mask: mask for grasping targets

        Returns:
            tuple(np.ndarray, np.ndarray): the 6-DoF grasping pose in the world base 
                the first element: an np.ndarray with the size of [3] and dtype of np.float64, representing the target position of the end effector
                the second element: an np.ndarray with the size of [4] and dtype of np.float64, representing the orientation of the pose
                    this element is a quarternion representing the rotation between the target pose and (1,0,0) (pointing forward, the default orientation of end effector)

        Notes: the axis direction of the world base are:
            x -> forward
            y -> left
            z -> upward
        """
        
        end_points, cloud = self.get_and_process_data(point_cloud, workspace_mask)
        gg = self.get_grasps(self.net, end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        gg.nms()
        gg.sort_by_score()
        # self.vis_grasps(gg, cloud)
        if len(gg) == 0:
            return (None, None), (None, cloud)
        translation = gg[0].translation
        quaternion = rota_to_quat(gg[0].rotation_matrix)
        vis = (gg,cloud)
        # translation[3], quaternion[4]
        return (translation, quaternion),vis

    def get_net(self):
        # Init the model
        net = graspnet(input_feature_dim=0,
                       num_angle=12,
                       num_depth=4,
                       cylinder_radius=0.05,
                       hmin=-0.02,
                       hmax_list=[0.01, 0.02, 0.03, 0.04],
                       is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self, input_cloud, workspace_mask):
        # generate cloud
        cloud = input_cloud[:, :, 0:3]
        color = input_cloud[:, :, 3:]
        # cloud_nomask = copy(cloud)
        # color_nomask = copy(color)
        cloud_nomask = cloud.reshape([-1, 3])
        color_nomask = color.reshape([-1, 3]) / 255.0

        # get valid points
        mask = workspace_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        cloud_masked = cloud_masked.reshape([-1, 3]).astype(np.float32)
        color_masked = color_masked.reshape([-1, 3]).astype(np.float32)

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_nomask.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_nomask.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud, top_k=5):
        gg = gg[:top_k]
        grippers = gg.to_open3d_geometry_list()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # Visualize the point cloud, grippers, and the coordinate frame
        # print(grippers)
        o3d.visualization.draw_geometries([cloud, *grippers, coordinate_frame])
    
    def vis_all_grasps(self, gg_list, cloud, top_k=5):
        gripper_list = []
        for gg in gg_list:
            if gg is None:
                continue
            gg = gg[:top_k]
            grippers = gg.to_open3d_geometry_list()
            gripper_list.extend(grippers)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([cloud, *gripper_list, coordinate_frame])

# @dataclass
# class AnyGraspConfig:
#     checkpoint_path: str
#     max_gripper_width: float
#     gripper_height: float
#     top_down_grasp: bool
#     debug: bool

# class AnyGrasp(GraspPredictor, model="anygrasp"):

#     def __init__(self, model) -> None:
#         super().__init__(model)
#         from gsnet import AnyGrasp
#         self.model = AnyGrasp(AnyGraspConfig(
#             checkpoint_path="/root/anygrasp_data/checkpoint_detection.tar",
#             max_gripper_width=0.060,
#             gripper_height=0.100,
#             top_down_grasp=False,
#             debug=True,
#         ))
#         self.model.load_net()

#     def infer(self, color_img: np.ndarray, depth_img: np.ndarray, mask: np.ndarray, camera_intrinsics: np.ndarray, shift: Tuple[np.ndarray, np.ndarray], end_pose: Tuple[np.ndarray, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#         colors = color_img.astype(np.float32) / 255.0
#         depths = depth_img.astype(np.float32)
#         xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
#         xmap, ymap = np.meshgrid(xmap, ymap)
#         cx = camera_intrinsics[0, 2]
#         cy = camera_intrinsics[1, 2]
#         fx = camera_intrinsics[0, 0]
#         fy = camera_intrinsics[1, 1]
#         points_x = (xmap - cx) / fx * depths
#         points_y = (ymap - cy) / fy * depths
#         points = np.stack([points_x, points_y, depths], axis=-1)

#         masked_pts = points[mask]
#         lims = [
#             masked_pts[:, 0].min(), masked_pts[:, 0].max(),
#             masked_pts[:, 1].min(), masked_pts[:, 1].max(),
#             masked_pts[:, 2].min(), masked_pts[:, 2].max(),
#         ]
#         lims[0] = np.clip(lims[0], -1.0, 1.0)
#         lims[1] = np.clip(lims[1], -1.0, 1.0)
#         lims[2] = np.clip(lims[2], -1.0, 1.0)
#         lims[3] = np.clip(lims[3], -1.0, 1.0)
#         lims[4] = np.clip(lims[4], 0.0, 2.0)
#         lims[5] = np.clip(lims[5], 0.0, 2.0)
#         print(f"AnyGrasp lims: {lims}")

#         points = points.astype(np.float32).reshape(-1, 3)
#         colors = colors.astype(np.float32).reshape(-1, 3)
#         pts_filter = (points[:, 2] > 0) & (points[:, 2] < 1)
#         points = points[pts_filter]
#         colors = colors[pts_filter]

#         sdk_result = self.model.get_grasp(points, colors, lims)
#         gg = sdk_result[0]
#         if gg is None or len(gg) == 0:
#             return None, None
#         gg = gg.nms().sort_by_score()
#         self.visualize_results(gg, sdk_result, lims)
#         translation = gg[0].translation
#         rotation_matrix = gg[0].rotation_matrix

#         T = pose2mat(end_pose) @ pose2mat(shift)
#         translation = T[:3, :3] @ translation + T[:3, 3]
#         rotation_matrix = T[:3, :3] @ rotation_matrix

#         return translation, rota_to_quat(rotation_matrix)

#     def visualize_results(self, gg, results, lims):
#         cloud=results[1]
#         grippers = gg.to_open3d_geometry_list()
#         grippers = grippers[:10]
#         lim_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(lims[0], lims[2], lims[4]), max_bound=(lims[1], lims[3], lims[5]))
#         o3d.visualization.draw_geometries([*grippers, cloud, lim_box])

if __name__ == '__main__':
    pass
