# Auther:guo xiao long
# @Time:2025/3/22 20:17
# @Author:gxl
# @site:找出数据集中没有打标签的图片，并删除
# @File:delete_imgs.py
# @software:PyCharm
import os
import time

# 设置目标目录路径（根据你的实际路径修改）
target_dir = r"F:\study\new_solder_data_test_onnx"
# 支持的图片格式（可扩展）
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
deleted_count = 0

start_time = time.time()

for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        file_name, file_ext = os.path.splitext(filename)
        if file_ext.lower() in image_extensions:
            # 生成对应的JSON文件名
            json_file = f"{file_name}.json"
            json_path = os.path.join(target_dir, json_file)
            # 检查JSON文件是否存在
            if not os.path.exists(json_path):
                try:
                    os.remove(file_path)
                    print(f"已删除无标签图片：{filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败：{filename}，错误原因：{str(e)}")

end_time = time.time()
print(f"执行完成，共删除 {deleted_count} 张图片，耗时：{end_time - start_time:.2f}秒")