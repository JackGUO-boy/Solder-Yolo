# Auther:guo xiao long
# @Time:2024/6/16 16:17
# @Author:gxl
# @site:
# @File:unet_dill.py
# @software:PyCharm
import warnings

warnings.filterwarnings('ignore')
import argparse, yaml, copy
# from ultralytics.models.yolo.detect.distill import DetectionDistiller
from ultralytics.models.yolo.segment.unet_dill  import UnetSegmentationDistiller
from ultralytics.models.yolo.pose.distill import PoseDistiller
from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'D:\yolov8\ultralytics-main\runs\segment\train\yolov8l_ppf2seg_dill(feauture)_adddata\weights\best.pt',
        'data': 'D:/yolov8/ultralytics_seg_dill/ultralytics//cfg/datasets/mydata.yaml',
        'imgsz': 640,
        'epochs': 220,
        'batch': 2,
        'workers': 2,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'project': 'runs/distill/dill_wrok_compare',
        'name': 'get_seg_result',

        # distill
        'prune_model': False,
        'teacher_weights': r'D:\yolov8\ultralytics-main\runs\segment\train2\yolov8l_add_dataset\weights\best.pt',
        # 'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8l-seg.yaml',
        'kd_loss_type': 'Unet_dill_loss',
        'kd_loss_decay': 'linear_epoch',

        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,

        'teacher_kd_layers': '9',  # '2,4,6,15,18',
        'student_kd_layers': '15',  # '2,4,6,10,14',
        'feature_loss_type': 'mimic',
        'feature_loss_ratio': 1.0
    }

    # model = DetectionDistiller(overrides=param_dict)
    model = UnetSegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()