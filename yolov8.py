import cv2
import os
import numpy as np
import yaml
from typing import Any
from ultralytics import YOLO
class YoloV8N:
    def __init__(self, model="yolov8n-seg.pt", cpu_only=False):
        
        # 
        self.yolo = YOLO(model)
        # self.yolo = YOLO("/root/workspace/weights/yolov8s-seg_best.pt")

        if not cpu_only:
            self.yolo = self.yolo.cuda()

        # self.coco_label = {41:"cup", 45:"bowl", 68:"microwave"}
        self.cup_idx=  41
        self.bowl_idx = 45
        # 初始化推理参数
        self.label = np.array([1])  # 标签1(前景点)或0(背景点)
        self.multimask_output = False
        print("YoloV8N initialized")

    def infer(self, image: np.ndarray, prompt: str, **kwargs: Any) -> dict:
        results = self.yolo.predict(image, verbose=False)
        bboxes = []
        confs = []
        masks = []
        # if prompt=="cup":
        #     label_idx = self.cup_idx
        # elif prompt == "bowl":
        #     label_idx = self.bowl_idx
        # else:
        #     label_idx = self.cup_idx

        if len(results) == 0:
            # num_obs = 0
            # print("NO boxes")
            return {"bbox": None, "text": prompt, "score": None, "mask": None}
        else:
            # label_name = results.names
            result = results[0]
            # print("bbox", result.boxes)
            detect_cls = result.boxes.cls.cpu().numpy().astype(int)
            detect_conf = result.boxes.conf.cpu().numpy()
            num_obs = detect_cls.shape[0]
            for i in range(num_obs):
                # if detect_cls[i] == label_idx:
                if prompt == "all":
                    confs.append(detect_conf[i])
                    x, y, w, h = result.boxes.xywh[i].cpu().numpy()
                    bbox = np.array([y, x, h, w]).astype(int)
                    bboxes.append(bbox)
                    mask_idx = np.array(result.masks.xy[i]).astype(int)
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask = cv2.fillPoly(mask, [mask_idx], color=(1, 1, 1))
                    masks.append(mask)
                else:
                    confs.append(detect_conf[i])
                    x, y, w, h = result.boxes.xywh[i].cpu().numpy()
                    bbox = np.array([y, x, h, w]).astype(int)
                    bboxes.append(bbox)
                    mask_idx = np.array(result.masks.xy[i]).astype(int)
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask = cv2.fillPoly(mask, [mask_idx], color=(1, 1, 1))
                    masks.append(mask)

            return {"bbox": bboxes, "text": prompt, "score": confs, "mask": masks}

    # def bbox_with_gap(self, bbox, gap):
    #     width = self.image.shape[1]
    #     height = self.image.shape[0]
    #     gapped_bbox = np.zeros(4)
    #     gapped_bbox[0] = np.maximum(0, bbox[0] - gap)
    #     gapped_bbox[1] = np.maximum(0, bbox[1] - gap)
    #     gapped_bbox[2] = np.minimum(width, bbox[2] + gap)
    #     gapped_bbox[3] = np.minimum(height, bbox[3] + gap)
    #     return bbox
