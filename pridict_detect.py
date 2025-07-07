# Auther:guo xiao long
# @Time:2025/3/25 10:27
# @Author:gxl
# @site:
# @File:pridict_detect.py
# @software:PyCharm
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO(r'D:\yolov8\ultralytics_seg_dill\runs\detect\train\weights\best.pt')

    # 预测参数配置
    results = model.predict(
        source=r'F:\study\Solder_Mask_Dataset_Yolo\images\test',  # 测试图片/视频路径（可替换为'0'使用摄像头）
        conf=0.25,  # 置信度阈值（根据需求调整）
        iou=0.7,  # IoU 阈值
        imgsz=640,  # 推理尺寸
        device=0,  # 使用GPU 0（CPU则改为'cpu'）
        save=True,  # 保存带标注结果
        show=False,  # 显示实时结果（批处理时自动关闭）
        show_labels=True,  # 显示标签
        show_conf=True  # 显示置信度
    )

    # # 可选：打印结果信息
    # for result in results:
    #     print("检测到对象：")
    #     print(f"类别: {result.names}")  # 输出类别名称映射
    #     print(f"检测框: {result.boxes.xyxy}")  # 输出边界框坐标
    #     print(f"置信度: {result.boxes.conf}")  # 输出置信度值