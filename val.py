# Auther:guo xiao long
# @Time:2024/5/29 17:57
# @Author:gxl
# @site:
# @File:val.py
# @software:PyCharm
from ultralytics import  YOLO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon
from ultralytics.utils.torch_utils import model_info
model = YOLO(r'D:\yolov8\ultralytics-main\runs\segment\train\yolov8l_ppf2seg_dill(feauture)_adddata\weights\best.pt')
img_path = r'D:\yolov8\ultralytics-main\datasets\Train_NG_OK\images\train\554.bmp'
results = model(img_path)
# 进行推理


# 获取第一个结果（如果处理单张图片）
result = results[0]
print(result.masks.data.shape)  # 打印掩码数据的形状
print(result.masks.data)
# 获取分割掩码
masks = result.masks.data  # 获取掩码数据
boxes = result.boxes  # 获取框数据

# 如果掩码数据是一个batch的，需要取出第一个
if masks.ndim == 3 and masks.shape[0] == 1:
    masks = masks[0]

# 加载原图像并获取尺寸
original_img = Image.open(img_path)
original_img_np = np.array(original_img)
height, width = original_img_np.shape[:2]

# 定义每个类别的颜色
colors = [
    [255, 0, 0],   # 类别0的颜色：红色
    [0, 255, 0],   # 类别1的颜色：绿色
    [0, 0, 255],   # 类别2的颜色：蓝色
    # 添加更多颜色以适应更多类别
]

# 获取每个框的类别
classes = boxes.cls.cpu().numpy().astype(int)

# 显示分割掩码
for i, mask in enumerate(masks):
    # 将Tensor转换为NumPy数组并进行类型转换
    mask_np = mask.cpu().numpy()  # 将Tensor转为NumPy数组
    mask_img = (mask_np * 255).astype(np.uint8)  # 将掩码转为8位图像

    # 调整掩码大小以匹配原始图像
    mask_img = Image.fromarray(mask_img).resize((width, height), Image.NEAREST)
    mask_img = np.array(mask_img)

    # 显示掩码图像
    plt.imshow(mask_img, cmap='gray')
    plt.title(f'Segmentation Mask {i+1}')
    plt.show()

# 可视化分割结果，将掩码叠加到原图像上
color_mask = np.zeros_like(original_img_np)
for i, mask in enumerate(masks):
    mask_np = mask.cpu().numpy()
    mask_img = (mask_np * 255).astype(np.uint8)

    # 调整掩码大小以匹配原始图像
    mask_img = Image.fromarray(mask_img).resize((width, height), Image.NEAREST)
    mask_img = np.array(mask_img)

    # 获取当前掩码的类别，并设置相应的颜色
    cls = classes[i]
    color = colors[cls % len(colors)]  # 使用模运算以防止颜色数组越界

    # 应用颜色到掩码区域
    color_mask[mask_img > 0] = color

# 叠加彩色掩码到原图像上
blended = Image.blend(original_img, Image.fromarray(color_mask), alpha=0.5)
plt.imshow(blended)
plt.title('Segmentation Result')
plt.show()

