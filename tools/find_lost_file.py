# Auther:guo xiao long
# @Time:2025/2/27 21:30
# @Author:gxl
# @site:查找遗漏的，未打标签的图片的名字
# @File:find_lost_file.py
# @software:PyCharm
import os


def find_images_without_labels(image_folder):
    # 获取文件夹中的所有文件
    all_files = os.listdir(image_folder)

    # 存储图像文件和标签文件的扩展名
    image_extensions = {'.jpg', '.png', '.bmp'}
    label_extension = '.json'

    images = []
    labels = set()

    # 遍历文件夹中的所有文件
    for file in all_files:
        file_name, ext = os.path.splitext(file)

        # 如果是图像文件
        if ext.lower() in image_extensions:
            images.append(file_name)

        # 如果是标签文件
        elif ext.lower() == label_extension:
            label_name = file_name  # 标签文件的文件名就是图像文件名
            labels.add(label_name)

    # 查找没有标签的图像
    missing_labels = [image + ext for image in images if image not in labels]

    return missing_labels


# 设置你的文件夹路径
image_folder = r'F:\study\new_solder_data_test_onnx'

# 查找没有标签的图像
missing_images = find_images_without_labels(image_folder)

# 打印结果
if missing_images:
    print("以下图片没有对应的标签文件:")
    for image in missing_images:
        print(image)
else:
    print("所有图片都有标签！")
