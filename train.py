# Auther:guo xiao long
# @Time:2024/5/28 18:32
# @Author:gxl
# @site:训练模型
# @File:train.py
# @software:PyCharm
import os
from ultralytics import  YOLO
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def main():
    model = YOLO(r'runs/segment/train/weights/last.pt')
    #model = YOLO(r'D:\yolov8\ultralytics-main\weights/yolov8n-seg.pt')
    #model.train(data=r'D:\yolov8\ultralytics-main\ultralytics\cfg\datasets\coco128-seg.yaml',epochs=300,imgsz=640,batch=4)
    model.train(data=r'D:\yolov8\ultralytics-main\ultralytics\cfg\datasets\coco128-seg.yaml', epochs=300, resume=True)


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()
#
#     # 查看模型参数量
#
#     #蒸馏前``
#
#
# from ultralytics import  YOLO
# from shapely.geometry import Polygon
# from ultralytics.utils.torch_utils import model_info
# model = YOLO(r'D:\yolov8\ultralytics_seg_dill\ultralytics\cfg\models\v8\yolov8x-seg.yaml')
# metrics = model.val(data=r"D:\yolov8\ultralytics_seg_dill\ultralytics\cfg\datasets\mydata.yaml")  # 在验证集上评估模型性能