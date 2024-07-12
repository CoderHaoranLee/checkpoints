#!/usr/bin/env python3
import os
os.environ['LM_CONFIG'] = "/root/workspace/weights/local.yaml"
os.environ['CKPT_DIR'] = '/root/workspace/weights/ckpt'

from typing import Tuple
import numpy as np
import rosbag
import cv2
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from graspmodel import GraspNet, pose2mat
from draw import draw_bbox, obb2poly
from scipy.spatial.transform import Rotation

import rospy
import copy
from yolov8 import YoloV8N

class GraspDetector:
    def __init__(self) -> None:
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback)
        self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback)
        self.depth_camera_info  = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self._camera_info_callback)
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.grasper = GraspNet(model='graspnet')
        self.yolo = YoloV8N()
        self.camera_shift = (
                        np.array([0.,  0. ,  0.]),
                        np.array([0.,  0.,  0., 1.0])
                    )
        self.arm_end_pose = (
                        np.array([ 0.0,  0.0, 0.0]),
                        np.array([ 0.0,  0.0, 0.0,  1.0])
                    )

    def _image_callback(self, msg):
        self.color_image = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _depth_callback(self, msg):
        self.depth_image = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _camera_info_callback(self, msg):
        # print("receive camera info")
        if self.camera_info is None:
            self.camera_info = msg
    

    def depth2cloud(self,depth_im, intrinsic_mat, organized=True):
        """ Generate point cloud using depth image only.
            Input:
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    depth image
                camera_info: dict

            Output:
                cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                    generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        height, width = depth_im.shape
        fx, fy, cx, cy = intrinsic_mat[0][0], intrinsic_mat[1][1], intrinsic_mat[0][2], intrinsic_mat[1][2]
        assert (depth_im.shape[0] == height and depth_im.shape[1] == width)
        xmap = np.arange(width)
        ymap = np.arange(height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_im  # change the unit to metel
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        return cloud

    def camera2base(self, camera_data, shift: Tuple[np.ndarray, np.ndarray], pose: tuple):
        """
                pose : tuple(array([, , ]), array([,  ,  ,  ]))
            """
        t = (pose2mat(pose) @ pose2mat(shift))

        origin_shape = camera_data.shape
        if origin_shape[-1] > 3:
            feat = camera_data[..., 3:]
        camera_data = camera_data[..., :3].reshape(-1, 3)
        camera_data_homogeneous = np.hstack((camera_data, np.ones((camera_data.shape[0], 1))))
        base_data = (t @ camera_data_homogeneous.T).T[:, :3]
        base_data = base_data.reshape(*origin_shape[:-1], -1)
        if origin_shape[-1] > 3:
            base_data = np.concatenate([base_data, feat], axis=-1)
        return base_data

    def base_cloud(self, image, depth, intrinsic, shift, end_pose):
        cam_cloud = self.depth2cloud(depth, intrinsic)
        cam_cloud = np.copy(np.concatenate((cam_cloud, image), axis=2))
        return self.camera2base(cam_cloud, shift, end_pose)
    
    def detector(self, top_k=5):
        if self.camera_info is not None:
            camera_intrinsics = np.array(self.camera_info.K).reshape(3,3)
            if self.color_image is not None and self.depth_image is not None:
                color = copy.deepcopy(self.color_image)
                depth = copy.deepcopy(self.depth_image) / 1000.0
                depth[depth>2.0] = 2.0
                print(np.min(depth), np.mean(depth), np.max(depth), np.median(depth))
                cloud = self.base_cloud(color, depth, camera_intrinsics, self.camera_shift, self.arm_end_pose)
                # det_results = self.yolo.infer(image=color, prompt="all")
                # bboxes = det_results["bbox"]
                # masks = det_results["mask"]

                # bboxes = det_results["bbox"]
                # masks = det_results["mask"]
                # image_draw = color
                # image_draw = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)
                # score = [1.]
                # if bboxes is not None and len(bboxes)>0:
                #     # merged_mask = np.zeros_like(masks[0])
                #     # for mask in masks:
                #     #     merged_mask = np.logical_or(merged_mask, mask)
                #     # merged_mask = merged_mask[:, :, np.newaxis]
                #     # image_draw = image_draw * (merged_mask.astype(np.uint8) * 0.75 + 0.25)
                #     image_draw = draw_bbox(image_draw, obb2poly(bboxes).astype(int))
                #     image_draw = image_draw.astype(np.uint8)
                #     cv2.putText(image_draw,
                #                 f"score: {score}",
                #                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.imshow('RGB', image_draw)
                # cv2.waitKey(0)
                # grippers = []
                # vis_cloud = copy.deepcopy(cloud)
                # for bbox_mask in masks:
                #     print("graspnet inference ...")
                #     (grasp_position, grasp_rotation), vis = self.grasper.infer(cloud, bbox_mask>0)

                #     print("visualizition ...")
                #     grippers.append(vis[0])
                #     vis_cloud = vis[1]
                #     # grasper.vis_grasps(vis[0], vis[1], top_k=10)
                #     # cv2.imshow('RGB', image_draw)
                #     # cv2.waitKey(0)
                # self.grasper.vis_all_grasps(grippers, vis_cloud, top_k=10)
                bbox_mask = np.ones_like(color[:, :, 0], dtype=bool)
                # print("graspnet inference ...")
                (grasp_position, grasp_rotation), vis = self.grasper.infer(cloud, bbox_mask)
                print(grasp_position, grasp_rotation)
                # print("visualizition ...")
                self.grasper.vis_grasps(vis[0], vis[1], top_k=top_k)
            else:
                print("color or depth is not ready!")
        else:
            print("camera_info is not ready!")
if __name__ == "__main__":
    rospy.init_node("graspnet_detector")
    grasp_detector = GraspDetector()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print("detection ...")
        grasp_detector.detector(top_k=100)
        rate.sleep()
    rospy.spin()
