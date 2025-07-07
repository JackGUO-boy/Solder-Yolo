# Auther:guo xiao long
# @Time:2025/3/25 13:29
# @Author:gxl
# @site:
# @File:develop_onnx_vedio.py
# @software:PyCharm
#coding:utf-8
import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import os
from scipy.optimize import linear_sum_assignment
import glob
# 置信度
confidence_thres = 0.35
# iou阈值
iou_thres = 0.5
# 类别
# classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
#            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
#            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
#            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
#            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
#            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
#            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
#            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
#            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
#            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
#            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
classes = {0: 'Circular_joint', 1: 'Cu', 2: 'Deviation', 3: 'Fault_joint', 4: 'Missing_Solder'}
# 随机颜色
color_palette = np.random.uniform(100, 255, size=(len(classes), 3))

# 判断是使用GPU或CPU
providers =['CPUExecutionProvider']
def calculate_iou_matrix(pred_boxes, true_boxes):

    N = len(pred_boxes)
    M = len(true_boxes)
    iou_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = calculate_iou(pred_boxes[i], true_boxes[j])
    return iou_matrix


def match_boxes(iou_matrix):
    """
    使用匈牙利算法匹配预测框和真实框。

    参数:
        iou_matrix (np.ndarray): IoU 矩阵，形状为 [N, M]。

    返回:
        tuple: (matched_pred_indices, matched_true_indices)，匹配的预测框和真实框的索引。
    """
    # 使用匈牙利算法找到最大匹配
    pred_indices, true_indices = linear_sum_assignment(-iou_matrix)  # 取负值，因为算法是最小化
    return pred_indices, true_indices
def calculate_matched_iou(pred_boxes, true_boxes, pred_class_ids, true_class_ids):
    """
    计算匹配后的IoU列表，仅当类别ID匹配时计算。

    参数:
        pred_boxes (list): 预测的边界框列表，每个框格式为 [x_center, y_center, width, height]。
        pred_class_ids (list): 预测的类别ID列表。
        true_boxes (list): 真实的边界框列表，每个框格式为 [x_center, y_center, width, height]。
        true_class_ids (list): 真实的类别ID列表。

    返回:
        list: 匹配后的IoU值列表。
    """
    # 转换为 numpy 数组
    pred_boxes = np.array(pred_boxes)
    true_boxes = np.array(true_boxes)

    # 计算 IoU 矩阵
    iou_matrix = calculate_iou_matrix(pred_boxes, true_boxes)

    # 匹配框
    pred_indices, true_indices = match_boxes(iou_matrix)

    # 计算匹配后的 IoU
    iou_list = []
    for pred_idx, true_idx in zip(pred_indices, true_indices):
        # 仅当类别ID匹配时计算IoU
        if pred_class_ids[pred_idx] == true_class_ids[true_idx]:
            iou = iou_matrix[pred_idx, true_idx]
            iou_list.append(iou)
    return iou_list
def calculate_iou(box, other_boxes):
    """
    计算给定边界框与一组其他边界框之间的交并比（IoU）。

    参数：
    - box: 单个边界框，格式为 [x1, y1, width, height]。
    - other_boxes: 其他边界框的数组，每个边界框的格式也为 [x1, y1, width, height]。

    返回值：
    - iou: 一个数组，包含给定边界框与每个其他边界框的IoU值。
    """
    # 确保 other_boxes 是二维数组
    other_boxes = np.array(other_boxes)
    if other_boxes.ndim == 1:
        other_boxes = np.expand_dims(other_boxes, axis=0)  # 转换为一维数组为二维
    # 计算交集的左上角坐标
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    # 计算交集的右下角坐标
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    # 计算交集区域的面积
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # 计算给定边界框的面积
    box_area = box[2] * box[3]
    # 计算其他边界框的面积
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    # 计算IoU值
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou
def load_true_boxes(label_path, img_width, img_height):
    """
    从标签文件中加载真实框。

    参数：
    - label_path: 标签文件的路径。

    返回值：
    - true_boxes: 真实框列表，格式为 [[x1, y1, width, height], ...]。
    """
    true_boxes = []
    true_class_ids = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                # 计算左上角坐标
                left = x_center - width / 2
                top = y_center - height / 2
                # 添加到列表中
                true_boxes.append([left, top, width, height])
                true_class_ids.append(class_id)
    return true_boxes, true_class_ids


def read_yolo_labels(label_path):
    """
    读取 YOLO 格式的标签文件。

    参数：
        label_path (str): 标签文件路径。

    返回：
        list: 包含所有边界框的列表，每个边界框格式为 [x1, y1, width, height]。
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_center, y_center, w, h = map(float, parts)
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            boxes.append([x1, y1, w, h])
    return boxes
def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    # 如果没有边界框，则直接返回空列表
    if len(boxes) == 0:
        return []
    # 将得分和边界框转换为NumPy数组
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 根据置信度阈值过滤边界框
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    # 如果过滤后没有边界框，则返回空列表
    if len(filtered_boxes) == 0:
        return []
    # 根据置信度得分对边界框进行排序
    sorted_indices = np.argsort(filtered_scores)[::-1]
    # 初始化一个空列表来存储选择的边界框索引
    indices = []
    # 当还有未处理的边界框时，循环继续
    while len(sorted_indices) > 0:
        # 选择得分最高的边界框索引
        current_index = sorted_indices[0]
        indices.append(current_index)
        # 如果只剩一个边界框，则结束循环
        if len(sorted_indices) == 1:
            break
        # 获取当前边界框和其他边界框
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        # 计算当前边界框与其他边界框的IoU
        iou = calculate_iou(current_box, other_boxes)
        # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        # 更新sorted_indices以仅包含不重叠的边界框
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    # 返回选择的边界框索引
    return indices


def draw_detections(img, box, score, class_id):
    """
    在输入图像上绘制检测到的对象的边界框和标签。

    参数:
            img: 要在其上绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。

    返回:
            无
    """

    # 提取边界框的坐标
    x1, y1, w, h = box
    # 根据类别ID检索颜色
    color = color_palette[class_id]
    # 在图像上绘制边界框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    # 创建标签文本，包括类名和得分
    label = f'{classes[class_id]}: {score:.2f}'
    # 计算标签文本的尺寸
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 计算标签文本的位置
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    # 绘制填充的矩形作为标签文本的背景
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    # 在图像上绘制标签文本
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def preprocess(img, input_width, input_height):
    """
    在执行推理之前预处理输入图像。

    返回:
        image_data: 为推理准备好的预处理后的图像数据。
    """

    # 获取输入图像的高度和宽度
    img_height, img_width = img.shape[:2]
    # 将图像颜色空间从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像大小调整为匹配输入形状
    img = cv2.resize(img, (input_width, input_height))
    # 通过除以255.0来归一化图像数据
    image_data = np.array(img) / 255.0
    # 转置图像，使通道维度为第一维
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道首
    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    # 返回预处理后的图像数据
    return image_data, img_height, img_width

def postprocess(input_image, output, input_width, input_height, img_width, img_height):
    """
    对模型输出进行后处理，提取边界框、得分和类别ID。

    参数:
        input_image (numpy.ndarray): 输入图像。
        output (numpy.ndarray): 模型的输出。
        input_width (int): 模型输入宽度。
        input_height (int): 模型输入高度。
        img_width (int): 原始图像宽度。
        img_height (int): 原始图像高度。

    返回:
        dict: 包含检测结果的字典，包括边界框、得分和类别ID。
    """
    # 转置和压缩输出以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))
    # 获取输出数组的行数
    rows = outputs.shape[0]
    # 用于存储检测的边界框、得分和类别ID的列表
    boxes = []
    scores = []
    class_ids = []
    # 计算边界框坐标的缩放因子
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    # 遍历输出数组的每一行
    for i in range(rows):
        # 从当前行提取类别得分
        classes_scores = outputs[i][4:]
        # 找到类别得分中的最大得分
        max_score = np.amax(classes_scores)
        # 如果最大得分高于置信度阈值
        if max_score >= confidence_thres:
            # 获取得分最高的类别ID
            class_id = np.argmax(classes_scores)
            # 从当前行提取边界框坐标
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            # 计算边界框的缩放坐标
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            # 将类别ID、得分和框坐标添加到各自的列表中
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    # 应用非最大抑制过滤重叠的边界框
    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    # 遍历非最大抑制后的选定索引
    for i in indices:
        # 根据索引获取框、得分和类别ID
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        # 在输入图像上绘制检测结果
        draw_detections(input_image, box, score, class_id)
    # 返回包含检测结果的字典
    return {
        'boxes': [boxes[i] for i in indices],
        'scores': [scores[i] for i in indices],
        'class_ids': [class_ids[i] for i in indices],
        'image': input_image
    }

def init_detect_model(model_path):
    # 使用ONNX模型文件创建一个推理会话，并指定执行提供者
    session = ort.InferenceSession(model_path, providers=providers)
    # 获取模型的输入信息
    model_inputs = session.get_inputs()
    # 获取输入的形状，用于后续使用
    input_shape = model_inputs[0].shape
    # 从输入形状中提取输入宽度
    input_width = input_shape[2]
    # 从输入形状中提取输入高度
    input_height = input_shape[3]
    # 返回会话、模型输入信息、输入宽度和输入高度
    return session, model_inputs, input_width, input_height

def detect_object(image, session, model_inputs, input_width, input_height):
    # 如果输入的图像是PIL图像对象，将其转换为NumPy数组
    if isinstance(image, Image.Image):
        result_image = np.array(image)
    else:
        # 否则，直接使用输入的图像（假定已经是NumPy数组）
        result_image = image
        # 预处理图像数据，调整图像大小并可能进行归一化等操作
    img_data, img_height, img_width = preprocess(result_image, input_width, input_height)
    # 使用预处理后的图像数据进行推理
    outputs = session.run(None, {model_inputs[0].name: img_data})
    # 对推理结果进行后处理，例如解码检测框，过滤低置信度的检测等
    result = postprocess(result_image, outputs, input_width, input_height, img_width, img_height)
    # 返回检测结果
    return result
def calculate_iou_for_boxes(pred_boxes, pred_class_ids, true_boxes, true_class_ids):
    """
    计算预测框和真实框之间的IoU列表，仅当类别ID匹配时计算。

    参数:
        pred_boxes (list): 预测的边界框列表，每个框格式为 [x_center, y_center, width, height]。
        pred_class_ids (list): 预测的类别ID列表。
        true_boxes (list): 真实的边界框列表，每个框格式为 [x_center, y_center, width, height]。
        true_class_ids (list): 真实的类别ID列表。

    返回:
        list: IoU 值列表。
    """
    iou_list = []
    for pred_box, pred_class_id in zip(pred_boxes, pred_class_ids):
        for true_box, true_class_id in zip(true_boxes, true_class_ids):
            # 仅当类别ID匹配时计算IoU
            if pred_class_id == true_class_id:
                iou = calculate_iou(pred_box, true_box)
                iou_list.append(iou)
    return iou_list



def process_folder(image_folder_path, label_folder_path, session, model_inputs, input_width, input_height):
    """
    处理文件夹中的所有图片，进行预测并计算指标。

    参数:
        image_folder_path: 图片文件夹路径。
        label_folder_path: 标签文件夹路径。
        session: ONNX推理会话。
        model_inputs: 模型输入信息。
        input_width: 模型输入宽度。
        input_height: 模型输入高度。

    返回:
        metrics: 包含mIoU、最大IoU、最小IoU、平均预测时间和总处理时间的字典。
    """
    total_iou = 0
    max_iou = 0
    min_iou = 1
    total_time = 0
    total_images = 0
    total_true_boxes = 0  # 用于统计真实框的总数

    # 遍历图片文件夹中的所有图片
    for image_name in os.listdir(image_folder_path):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder_path, image_name)
            label_path = os.path.join(label_folder_path, os.path.splitext(image_name)[0] + '.txt')

            # 读取图片
            image_data = cv2.imread(image_path)
            if image_data is None:
                print(f"无法读取图片: {image_path}")
                continue

            # 获取图像宽度和高度
            img_height, img_width, _ = image_data.shape

            # 加载真实框并反归一化
            true_boxes, true_class_ids = load_true_boxes(label_path, img_width, img_height)
            total_true_boxes += len(true_boxes)  # 累加真实框的数量

            # 进行预测
            start_time = time.time()
            result = detect_object(image_data, session, model_inputs, input_width, input_height)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            total_images += 1

            # 提取预测框和类别ID
            pred_boxes = result['boxes']
            pred_class_ids = result['class_ids']

            # 计算匹配后的IoU
            if pred_boxes and true_boxes:
                iou_list = calculate_matched_iou(pred_boxes, true_boxes, pred_class_ids, true_class_ids)
                if iou_list:
                    total_iou += sum(iou_list)
                    max_iou = max(max_iou, max(iou_list))
                    min_iou = min(min_iou, min(iou_list))

    # 所有图像处理完成后，计算mIoU
    mIoU = total_iou / total_true_boxes if total_true_boxes > 0 else 0

    # 计算指标
    metrics = {
        'mIoU': mIoU,
        'max_iou': max_iou,
        'min_iou': min_iou,
        'avg_time': total_time / total_images if total_images > 0 else 0,
        'total_time': total_time
    }
    return metrics
if __name__ == '__main__':
    # 模型文件的路径
    model_path = r"D:\yolov8\ultralytics_seg_dill\runs\detect\train_yolov8_detect\weights\best.onnx"
    # 初始化检测模型，加载模型并获取模型输入节点信息和输入图像的宽度、高度
    session, model_inputs, input_width, input_height = init_detect_model(model_path)
    # 四种模式 0为文件夹图片预测；1为图片预测，并显示结果图片；2为摄像头检测，并实时显示FPS； 3为视频检测，并保存结果视频
    image_folder_path = r"F:\study\Solder_Mask_Dataset_Yolo\images\test"
    label_folder_path=r"F:\study\Solder_Mask_Dataset_Yolo\labels\test"
    mode = 0
    if mode==0:
        # 处理文件夹中的图片并计算指标
        metrics = process_folder(image_folder_path, label_folder_path, session, model_inputs, input_width, input_height)
        # 输出结果
        print(f"mIoU: {metrics['mIoU']}")
        print(f"最大IoU: {metrics['max_iou']}")
        print(f"最小IoU: {metrics['min_iou']}")
        print(f"平均预测时间: {metrics['avg_time'] * 1000} ms")
        print(f"总处理时间: {metrics['total_time']} s")

    elif mode == 1:
        # 读取图像文件
        image_data = cv2.imread(r"F:\study\Solder_Mask_Dataset_Yolo\images\test/20250227093033_0.jpg.jpg")
        # 使用检测模型对读入的图像进行对象检测
        start_time = time.time()
        result_image = detect_object(image_data, session, model_inputs, input_width, input_height)

        end_time = time.time()
        print("推理时间:", (end_time - start_time) * 1000)
        # 将检测后的图像保存到文件
        cv2.imwrite("output_image.jpg", result_image)
        # 在窗口中显示检测后的图像
        #cv2.imshow('Output', result_image)
        # 等待用户按键，然后关闭显示窗口
        cv2.waitKey(0)
    elif mode == 2:
        # 打开摄像头
        cap = cv2.VideoCapture()  # 0表示默认摄像头，如果有多个摄像头可以尝试使用1、2等
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        # 初始化帧数计数器和起始时间
        frame_count = 0
        start_time = time.time()
        # 循环读取摄像头视频流
        while True:
            # 读取一帧
            ret, frame = cap.read()
            # 检查帧是否成功读取
            if not ret:
                print("Error: Could not read frame.")
                break
            # 使用检测模型对读入的帧进行对象检测
            output_image = detect_object(frame, session, model_inputs, input_width, input_height)
            # 计算帧速率
            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            # 将FPS绘制在图像上
            cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # 在窗口中显示当前帧
            cv2.imshow("Video", output_image)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放摄像头资源
        cap.release()
        # 关闭窗口
        cv2.destroyAllWindows()
    elif mode == 3:
        # 输入视频路径
        input_video_path = 'kun.mp4'
        # 输出视频路径
        output_video_path = 'kun_det.mp4'
        # 打开视频文件
        cap = cv2.VideoCapture(input_video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        # 读取视频的基本信息
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 定义视频编码器和创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 根据文件名后缀使用合适的编码器
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        # 初始化帧数计数器和起始时间
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Info: End of video file.")
                break
            # 对读入的帧进行对象检测
            output_image = detect_object(frame, session, model_inputs, input_width, input_height)
            # 计算并打印帧速率
            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
            # 将处理后的帧写入输出视频
            out.write(output_image)
            #（可选）实时显示处理后的视频帧
            cv2.imshow("Output Video", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("输入错误，请检查mode的赋值")

