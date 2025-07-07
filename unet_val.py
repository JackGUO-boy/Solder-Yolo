# Auther:guo xiao long
# @Time:2024/6/27 15:07
# @Author:gxl
# @site:
# @File:unet_val.py
# @software:PyCharm
# 加载预训练的UNet模型
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ultralytics.models.unet.unet import Unet
model_path = r'D:\UNet_Demo\logs\best_epoch_weights.pth'  # 替换为实际的模型路径
model = Unet(num_classes=3, backbone="vgg")
model.load_state_dict(torch.load(model_path))
model.eval()

# 定义图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整为模型输入的尺寸
    transforms.ToTensor(),
])


# 处理单张图片并生成分割效果图
def process_image(img_path, model, preprocess, name_classes):
    # 加载图像并预处理
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度

    # 进行推理
    with torch.no_grad():
        output = model(input_tensor)

    # 获取分割掩码
    output = torch.sigmoid(output)
    output_np = output.squeeze().cpu().numpy()

    # 定义颜色
    colors = [
        [0, 0, 0],  # 类别0的颜色：黑色（背景）
        [255, 0, 0],  # 类别1的颜色：红色
        [0, 255, 0],  # 类别2的颜色：绿色
    ]

    # 加载原图像并获取尺寸
    original_img = Image.open(img_path)
    original_img_np = np.array(original_img)
    height, width = original_img_np.shape[:2]

    # 创建彩色掩码
    color_mask = np.zeros_like(original_img_np)

    # 处理每个通道的掩码
    for cls in range(output_np.shape[0]):
        mask_img = (output_np[cls] > 0.5).astype(np.uint8) * 255  # 二值化掩码
        mask_img_resized = Image.fromarray(mask_img).resize((width, height), Image.NEAREST)
        mask_img_resized = np.array(mask_img_resized)

        # 应用颜色到掩码区域
        color_mask[mask_img_resized > 0] = colors[cls]

    # 叠加彩色掩码到原图像上
    blended = Image.blend(original_img, Image.fromarray(color_mask), alpha=0.5)

    # 返回分割结果和叠加效果图
    return color_mask, blended


# 设置图片文件夹路径和预测结果保存文件夹路径
img_folder = r'D:\yolov8\ultralytics-main\datasets\Train_NG_OK\images\train'
predict_folder = r'D:\yolov8\ultralytics_seg_dill\runs\unet_predict'

# 确保预测结果保存文件夹存在
if not os.path.exists(predict_folder):
    os.makedirs(predict_folder)

# 获取所有图片文件的路径
img_files = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if
             img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
name_classes = ["_background_", "null", "white_area"]
# 批量处理图片
for img_path in img_files:
    try:
        # 处理图片并获取分割结果和叠加效果图
        color_mask, blended = process_image(img_path, model, preprocess, name_classes)

        # 获取图片的原始名称和扩展名
        img_name = os.path.basename(img_path)
        img_name_without_ext, img_ext = os.path.splitext(img_name)

        # 定义保存预测结果的文件路径
        predict_img_path = os.path.join(predict_folder, f'{img_name_without_ext}_predict{img_ext}')
        blended.save(predict_img_path)

        print(f'Predicted image saved as: {predict_img_path}')

    except Exception as e:
        print(f'Error processing image {img_path}: {e}')

# 显示最后处理的一张图片的分割结果
plt.imshow(blended)
plt.title('Segmentation Result')
plt.show()