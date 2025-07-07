# Auther:guo xiao long
# @Time:2025/3/25 10:21
# @Author:gxl
# @site:
# @File:val_detect.py
# @software:PyCharm
from ultralytics import  YOLO
from ultralytics import YOLO

if __name__ == '__main__':
    # 初始化模型
    model = YOLO(r'D:\yolov8\ultralytics_seg_dill\runs\detect\train\weights\best.pt')

    # 验证配置（添加 workers=0 可临时禁用多进程）
    metrics = model.val(
        data=r"D:\yolov8\ultralytics_seg_dill\ultralytics\cfg\datasets\Detect_Solder.yaml",
        batch=8,  # 根据GPU显存调整
        imgsz=640,  # 训练使用的图像尺寸
        conf=0.3,  # 置信度阈值
        iou=0.5,  # IoU阈值
        device=0,  # 使用GPU 0
        save_json=False,  # 生成JSON结果
        # workers=0     # 若仍有问题可取消注释此行（禁用多进程加载）
    )