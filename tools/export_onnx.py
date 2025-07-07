# Auther:guo xiao long
# @Time:2024/9/5 16:24
# @Author:gxl
# @site:导出为ONNX
# @File:export_onnx.py
# @software:PyCharm
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO(r"D:\yolov8\ultralytics_seg_dill\runs\detect\train_yolov8_detect\weights\best.pt")

# Export the model,参数opset为导出版本，dynamic=True表示输入图片是否可以是动态的，反之则限制为一定尺寸，imgsz即是限定尺寸为640x640（没有onnx就pip install onnx）
model.export(format="onnx",imgsz=640)# ,int8=True