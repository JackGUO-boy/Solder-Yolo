# Auther:guo xiao long
# @Time:2024/5/28 18:32
# @Author:gxl
# @site:知识蒸馏
# @File:train.py
# @software:PyCharm
import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
#from ultralytics.models.yolo.detect.distill import DetectionDistiller
from ultralytics.models.yolo.segment.distill import SegmentationDistiller
from ultralytics.models.yolo.pose.distill import PoseDistiller
from ultralytics.models.yolo.obb.distill import OBBDistiller


def freeze_model(trainer):
    # Retrieve the batch data
    model = trainer.model
    print('Befor Freeze')
    for k, v in model.named_parameters():
        print('\t', k, '\t', v.requires_grad)

    # freeze = 10
    # freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    freeze = 'model.12.' # 冻结指定层 从0开始
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print('After Freeze')
    for k, v in model.named_parameters():
        print('\t', k, '\t', v.requires_grad)






if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'D:\yolov8\ultralytics_seg_dill\ultralytics\cfg\models\v8\student2_Attention.yaml',
        #'model': r"D:\yolov8\ultralytics_seg_dill\runs\segment\10000_img_new_train_result\v8x_2_v8l\weights/last.pt",
        'data':r'D:\yolov8\ultralytics-main\ultralytics\cfg\datasets\coco128-seg.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 4,
        'workers': 2,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'project':'runs/segment/10000_img_new_train_result',
        'name':'v8l_2_stu_cbam',
        'amp' :False  ,
        'resume':False,#是否从断点继续训练



        # distill
        'prune_model': False,#模型剪枝
     #  'teacher_weights': r'D:\yolov8\ultralytics_seg_dill\add_train\runs\fist_step_dill\l_n_sgd_cwd_tau3\weights\best.pt',
       'teacher_weights': r'D:\yolov8\ultralytics_seg_dill\runs\segment\10000_img_new_train_result\v8x_2_v8l\weights\best.pt',

        #'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8l-seg.yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'linear_epoch',
        
        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '2,4,6,15,18',#'2,4,6,15,18', 256-512 v8x-v8l-->2,4,6,15,18,21
        'student_kd_layers': '2,4,6,10,15',#'2,4,6,10,15', 256-512,seg is 128  v8l-student--.2，4，6，15，18  2,4,6,8,11
        'feature_loss_type': 'cwd',
        'feature_loss_ratio':1 #mgd 调小 0.03  cwd 调1
    }
    

    model = SegmentationDistiller(overrides=param_dict)#是否从断点开始继续蒸馏


    # model.add_callback("on_pretrain_routine_start", freeze_model)
    model.distill(resume=True)