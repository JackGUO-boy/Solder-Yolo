# Auther:guo xiao long
# @Time:2025/3/22 20:27
# @Author:gxl
# @site:划分数据集（目标检测）
# @File:split_dataSet.py
# @software:PyCharm
import json
import os

import os
import json
import random
import shutil
from PIL import Image


def convert_xanylabeling_to_yolov8(json_path, output_dir, classes):
    """
    转换单个JSON标注文件为YOLOv8格式
    :param json_path: JSON文件路径
    :param output_dir: 输出目录
    :param classes: 类别列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_file = os.path.splitext(os.path.basename(json_path))[0] + '.jpg'
    image_path = os.path.join(os.path.dirname(json_path), image_file)
    txt_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + '.txt')

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except FileNotFoundError:
        print(f"警告：图片{image_file}不存在，已跳过")
        return

    with open(txt_file, 'w') as out_file:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # 确保是矩形标注（4个顶点）
            if len(points) != 4:
                continue

            # 计算边界框坐标
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            # 转换YOLO格式
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 获取类别索引
            class_idx = classes.index(label)

            out_file.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def split_dataset(input_dir, output_dir, ratios=(0.8, 0.1, 0.1)):
    """
    数据集划分与格式转换
    :param input_dir: 输入目录（包含图片和JSON文件）
    :param output_dir: 输出根目录
    :param ratios: 划分比例（train, val, test）
    """
    # 创建目录结构
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', subset), exist_ok=True)

    # 获取所有JSON文件并打乱顺序
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    random.shuffle(json_files)

    # 自动生成类别列表
    classes = set()
    for json_file in json_files:
        with open(os.path.join(input_dir, json_file), 'r') as f:
            data = json.load(f)
            for shape in data['shapes']:
                classes.add(shape['label'])
    classes = sorted(list(classes))

    # 保存类别文件
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes))

    # 划分数据集
    total = len(json_files)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    subsets = {
        'train': json_files[:train_end],
        'val': json_files[train_end:val_end],
        'test': json_files[val_end:]
    }

    # 处理每个子集
    for subset, files in subsets.items():
        for json_file in files:
            # 转换标签
            json_path = os.path.join(input_dir, json_file)
            convert_xanylabeling_to_yolov8(
                json_path,
                os.path.join(output_dir, 'labels', subset),
                classes
            )

            # 复制图片
            image_file = json_file.replace('.json', '.jpg')
            src_img = os.path.join(input_dir, image_file)
            dst_img = os.path.join(output_dir, 'images', subset, image_file)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)


if __name__ == '__main__':
    # 输入参数设置
    input_dir = r"F:\study\new_solder_data_test_onnx"  # 替换为输入目录
    output_dir = r"F:\study\Solder_Mask_Dataset_Yolo"  # 替换为输出目录

    # 执行转换和划分
    split_dataset(input_dir, output_dir)



