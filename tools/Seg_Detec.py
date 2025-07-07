# Auther:guo xiao long
# @Time:2025/3/26 11:05
# @Author:gxl
# @site:将图像进行分割--再利用掩膜进行检测
# @File:Seg_Detec.py
# @software:PyCharm
# Auther:guo xiao long
# @Time:2025/3/20 21:42
# @Author:gxl
# @site:
# @File:develop_onnx.py
# @software:PyCharm
import cv2
from PIL import Image
import numpy as np
import onnxruntime as ort
import time
import os
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
classes = {0: 'Solder_joint', 1: 'Null', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
           14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
           29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
           40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
           48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
           55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
           62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
           69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
           76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF0000', '00FF00', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class YOLOv8Seg:
    def __init__(self, onnx_model):

        # Build Ort session
        self.session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == 'tensor(float16)' else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # Load COCO class names
        self.classes = classes

        # Create color palette
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.35, iou_threshold=0.3, nm=32):
        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        boxes, segments, masks = self.postprocess(preds,
                                                  im0=im0,
                                                  ratio=ratio,
                                                  pad_w=pad_w,
                                                  pad_h=pad_h,
                                                  conf_threshold=conf_threshold,
                                                  iou_threshold=iou_threshold,
                                                  nm=nm)
        return boxes, segments, masks

    def preprocess(self, img):
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:

            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                           interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=True, save=False,stacking=False):#是否在原图上叠加结果掩膜

        # Draw rectangles and polygons
        if not stacking:
            im_canvas = np.zeros_like(im)
        else:
            im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            # cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            # cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))
            # 根据类别选择颜色
            if int(cls_) == 0:  # 第一个类别用红色
                color = (0, 0, 255)  # BGR格式的红色
            elif int(cls_) == 1:  # 第二个类别用绿色
                color = (0, 255, 0)  # BGR格式的绿色
            else:  # 其他类别用蓝色（可选）
                color = (255, 0, 0)  # BGR格式的蓝色
            cv2.fillPoly(im_canvas, np.int32([segment]), color)
            # draw bbox rectangle
            # cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            #               self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)

            # cv2.putText(im, f'{self.classes[cls_]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)

        # Mix image 图像和掩膜混合比例，实现掩膜透明效果
        # im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        # return im
        return im_canvas

    def load_labels(self,label_file, img_shape):
        labels = []
        with open(label_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = list(map(float, line.split()))
                cls = int(data[0])  # 类别
                polygon = np.array(data[1:]).reshape(-1, 2)  # 多边形坐标对
                # 将归一化的坐标转化为图像尺寸的实际坐标
                polygon[:, 0] *= img_shape[1]  # 宽
                polygon[:, 1] *= img_shape[0]  # 高
                labels.append((cls, polygon))
        return labels

    def calculate_iou(self,pred_polygon, gt_polygon,img_shape):


        pred_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        gt_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

        cv2.fillPoly(pred_mask, [np.int32(pred_polygon)], 1)
        cv2.fillPoly(gt_mask, [np.int32(gt_polygon)], 1)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        iou = intersection / union if union > 0 else 0

        return iou

    def calculate_map_iou(self,pred_boxes, pred_segments, gt_labels):
        ious = []
        matches = []

        for pred_box, pred_segment in zip(pred_boxes, pred_segments):
            max_iou = 0
            best_match = None
            for gt_cls, gt_polygon in gt_labels:
                iou = self.calculate_iou(pred_segment, gt_polygon,img.shape)
                if iou > max_iou:
                    max_iou = iou
                    best_match = gt_cls

            if max_iou > 0.5:  # IoU threshold for a match
                matches.append((pred_box[5], best_match, max_iou))  # (pred_cls, gt_cls, iou)
            ious.append(max_iou)

        # 计算mAP和mIoU
        if len(matches) > 0:
            map_score = np.mean([match[2] for match in matches if match[0] == match[1]])
        else:
            map_score = 0  # 当没有匹配时，mAP设为0

        if len(ious) > 0:
            miou = np.mean(ious)
        else:
            miou = 0  # 当没有iou记录时，mIoU设为0

        return map_score, miou

    def calculate_fps(self,total_inference_time, image_count):
        fps = image_count / total_inference_time
        return fps

    def draw_gt_labels(self, img, gt_labels, color=(0, 255, 0)):
        """
        在图片上绘制真实标签，使用绿色标记（或指定的颜色）。
        """
        # 防止看不清楚，暂时不绘制真实标签
        # for gt in gt_labels:
        #     cls_, polygon = gt[0], gt[1]  # 元组解包，获取类别号和多边形
        #     # 绘制真实标签的多边形
        #     cv2.polylines(img, [np.int32(polygon)], True, color, 2)
        #     # 在多边形的起始点上绘制类别标签
        #     cv2.putText(img, f"GT: {self.classes[cls_]}", (int(polygon[0][0]), int(polygon[0][1]) - 9),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img

class YOLOv8Det:
    def __init__(self, model_path, confidence_thres=0.35, iou_thres=0.5):
        self.classes = {0: 'Circular_joint', 1: 'Cu', 2: 'Deviation', 3: 'Fault_joint', 4: 'Missing_Solder'}
        self.color_palette = np.random.uniform(100, 255, size=(len(self.classes), 3))
        self.providers = ['CPUExecutionProvider']
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Initialize model
        self.session, self.model_inputs, self.input_width, self.input_height = self.init_detect_model(model_path)

    def init_detect_model(self, model_path):
        session = ort.InferenceSession(model_path, providers=self.providers)
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape
        input_width = input_shape[2]
        input_height = input_shape[3]
        return session, model_inputs, input_width, input_height

    def calculate_iou(self, box, other_boxes):
        other_boxes = np.array(other_boxes)
        if other_boxes.ndim == 1:
            other_boxes = np.expand_dims(other_boxes, axis=0)

        x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
        y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
        x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
        y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])

        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = box[2] * box[3]
        other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
        iou = intersection_area / (box_area + other_boxes_area - intersection_area)
        return iou

    def calculate_iou_matrix(self, pred_boxes, true_boxes):
        N = len(pred_boxes)
        M = len(true_boxes)
        iou_matrix = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                iou_matrix[i, j] = self.calculate_iou(pred_boxes[i], true_boxes[j])
        return iou_matrix

    def match_boxes(self, iou_matrix):
        pred_indices, true_indices = linear_sum_assignment(-iou_matrix)
        return pred_indices, true_indices

    def calculate_matched_iou(self, pred_boxes, true_boxes, pred_class_ids, true_class_ids):
        pred_boxes = np.array(pred_boxes)
        true_boxes = np.array(true_boxes)
        iou_matrix = self.calculate_iou_matrix(pred_boxes, true_boxes)
        pred_indices, true_indices = self.match_boxes(iou_matrix)

        iou_list = []
        for pred_idx, true_idx in zip(pred_indices, true_indices):
            if pred_class_ids[pred_idx] == true_class_ids[true_idx]:
                iou = iou_matrix[pred_idx, true_idx]
                iou_list.append(iou)
        return iou_list

    def load_true_boxes(self, label_path, img_width, img_height):
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
                    left = x_center - width / 2
                    top = y_center - height / 2
                    true_boxes.append([left, top, width, height])
                    true_class_ids.append(class_id)
        return true_boxes, true_class_ids

    def read_yolo_labels(self, label_path):
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

    def custom_NMSBoxes(self, boxes, scores, confidence_threshold, iou_threshold):
        if len(boxes) == 0:
            return []

        scores = np.array(scores)
        boxes = np.array(boxes)
        mask = scores > confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        if len(filtered_boxes) == 0:
            return []

        sorted_indices = np.argsort(filtered_scores)[::-1]
        indices = []

        while len(sorted_indices) > 0:
            current_index = sorted_indices[0]
            indices.append(current_index)
            if len(sorted_indices) == 1:
                break

            current_box = filtered_boxes[current_index]
            other_boxes = filtered_boxes[sorted_indices[1:]]
            iou = self.calculate_iou(current_box, other_boxes)
            non_overlapping_indices = np.where(iou <= iou_threshold)[0]
            sorted_indices = sorted_indices[non_overlapping_indices + 1]
        return indices

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f'{self.classes[class_id]}: {score:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img, input_width, input_height):
        img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_width, input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data, img_height, img_width

    def postprocess(self, input_image, output, input_width, input_height, img_width, img_height):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        x_factor = img_width / input_width
        y_factor = img_height / input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = self.custom_NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)

        return {
            'boxes': [boxes[i] for i in indices],
            'scores': [scores[i] for i in indices],
            'class_ids': [class_ids[i] for i in indices],
            'image': input_image
        }

    def detect_object(self, image):
        if isinstance(image, Image.Image):
            result_image = np.array(image)
        else:
            result_image = image

        img_data, img_height, img_width = self.preprocess(result_image, self.input_width, self.input_height)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        result = self.postprocess(result_image, outputs, self.input_width, self.input_height, img_width, img_height)
        return result

    def calculate_iou_for_boxes(self, pred_boxes, pred_class_ids, true_boxes, true_class_ids):
        iou_list = []
        for pred_box, pred_class_id in zip(pred_boxes, pred_class_ids):
            for true_box, true_class_id in zip(true_boxes, true_class_ids):
                if pred_class_id == true_class_id:
                    iou = self.calculate_iou(pred_box, true_box)
                    iou_list.append(iou)
        return iou_list

    def process_folder(self, image):

        image_data = image
        if image_data is None:
            print(f"无法读取掩膜图片:")
            return
        start_time = time.time()
        result = self.detect_object(image_data)
        end_time = time.time()
        inference_time = end_time - start_time
        return result
if __name__ == '__main__':

#=====================================分割模型初始化
    # 模型路径
    Seg_model_path = r"D:\yolov8\ultralytics_seg_dill\runs\segment\new_solder_data_train\yolov8l_train\weights\best.onnx"
    # 实例化模型
    seg_model = YOLOv8Seg(Seg_model_path)
    seg_conf = 0.5
    seg_iou = 0.5
#======================================目标检测模型初始化
    Detect_model_path = r"D:\yolov8\ultralytics_seg_dill\runs\detect\train_yolov8_detect\weights\best.onnx"
    # 创建YOLOv8Det实例

    Detect_conf = 0.35
    # iou阈值
    Detect_iou_thres = 0.5
    detector = YOLOv8Det(Detect_model_path,confidence_thres=Detect_conf,iou_thres=Detect_iou_thres)
#=======================================模式选择、其他参数
    mode = 1


#========================================mode===1
    imgpath = r"20231009111648289289941.bmp"#mode = 1时，图片预测路径
    is_save = True
    imgsavepath = "result.jpg"#mode = 1时，图d片预测结果保存路径
#===========================================执行两阶段检测
    if mode == 1:
        # opencv 读取图片
        img = cv2.imread(imgpath)
        # 推理
        start_time1 = time.time()
        boxes, segments, _ = seg_model(img, conf_threshold=seg_conf, iou_threshold=seg_iou)
        end_time = time.time()
        print("分割推理时间:", (end_time - start_time1) * 1000,"ms")
        # 画图
        if len(boxes) > 0:
            output_image = seg_model.draw_and_visualize(img, boxes, segments, vis=False, save=True, stacking=False)  # 生成结果掩膜
            start_time = time.time()
            metrics = detector.process_folder(output_image)
            end_time = time.time()
            print("检测推理时间:", (end_time - start_time) * 1000,"ms")
            print("总时间：",(end_time-start_time1)* 1000,"ms")
            cv2.imwrite(imgsavepath,metrics['image'])
        else:
            print("分割图像失败，没有检测到物体")


















    # if mode == 0:
    #     if folderpath:
    #         image_files = [f for f in os.listdir(folderpath) if f.endswith(('.png', '.jpg', '.bmp'))]
    #         #统计推理时间和程序耗时
    #         image_count = 0
    #         no_object_detect_cont=0#记录没有检测到目标情况的图像个数
    #         folder_start_time = time.time()
    #         total_inference_time = 0
    #
    #         # 统计 mIoU 和 Precision 的列表
    #         miou_list = []
    #         precision_list = []
    #
    #         for image_file in tqdm(image_files, desc="处理图片"):
    #             image_path = os.path.join(folderpath, image_file)
    #             file_name = os.path.basename(image_path)
    #             label_file = os.path.join(label_path, f"{os.path.splitext(image_file)[0]}.txt")  # 获取对应的标签文件
    #             if not os.path.exists(label_file):
    #                 print(f"标签文件 {label_file} 不存在，跳过该图片")
    #                 continue
    #             # opencv 读取图片
    #             img = cv2.imread(image_path)
    #             gt_labels = model.load_labels(label_file, img.shape)  # 加载对应的标签文件
    #
    #             # 推理
    #             start_time = time.time()
    #             boxes, segments, _ = model(img, conf_threshold=conf, iou_threshold=iou)
    #
    #             # if len(boxes)==0 or len(segments)==0:
    #             #     print("当前图片未检测到目标，pass")
    #             #     pass
    #             end_time = time.time()
    #             print(f"{file_name} 时间:{(end_time-start_time)*1000:.2f} ")
    #
    #             total_inference_time += (end_time-start_time)
    #             image_count +=1
    #             # 画图
    #             if len(boxes) > 0:
    #                 output_image = model.draw_and_visualize(img, boxes, segments, vis=False, save=True,stacking=False)
    #                 map_score, miou = model.calculate_map_iou(boxes, segments, gt_labels)
    #                 # print(f"mAP: {map_score:.3f}, mIoU: {miou:.3f}")
    #                 # 保存 mIoU 和 Precision
    #                 miou_list.append(miou)
    #                 precision_list.append(map_score)
    #             else:
    #                 output_image = img
    #                 miou_list.append(0.0)
    #                 precision_list.append(0.0)
    #                 no_object_detect_cont +=1
    #
    #             # 在同一张图上绘制gt标签（用绿色显示）
    #             if len(gt_labels) > 0:
    #                 output_image = model.draw_gt_labels(output_image, gt_labels, color=(0, 255, 0))
    #             if is_save_folder:
    #                 cv2.imwrite(save_folder_path+"/"+file_name+".jpg".format(image_count), output_image)
    #             if is_show:
    #                 cv2.imshow("seg", output_image)
    #         fps = model.calculate_fps(total_inference_time, image_count)
    #         folder_end_time = time.time()
    #
    #         # 计算平均 mIoU，最大 mIoU，最小 mIoU 和平均精确度 P
    #         avg_miou = np.mean(miou_list) if miou_list else 0
    #         max_miou = np.max(miou_list) if miou_list else 0
    #         min_miou = np.min(miou_list) if miou_list else 0
    #         avg_precision = np.mean(precision_list) if precision_list else 0
    #         print(f"文件夹完成检测,共检测了{image_count}张图片,fps:{fps:.2f},其中有{no_object_detect_cont}张图片未能检测到物体")
    #         print(f"平均 mIoU: {avg_miou:.3f}, 最大 mIoU: {max_miou:.3f}, 最小 mIoU: {min_miou:.3f}")
    #         print(f"平均精确度 P: {avg_precision:.3f}")
    #         print(f"程序总共耗时：{(folder_end_time-folder_start_time)*1000:.2f}ms")
    #         print(f"总推理时间：{total_inference_time*1000:.2f}ms")
    #         print(f"平均1张图片推理时间：{total_inference_time*1000/image_count:.2f}ms")
    #
    # elif mode == 1:
    #     # opencv 读取图片
    #     img = cv2.imread(imgpath)
    #     # 推理
    #     start_time = time.time()
    #     boxes, segments, _ = model(img, conf_threshold=conf, iou_threshold=iou)
    #     end_time = time.time()
    #     print("推理时间:",(end_time-start_time)*1000)
    #     # 画图
    #     if len(boxes) > 0:
    #         output_image = model.draw_and_visualize(img, boxes, segments, vis=False, save=True,stacking=False)#生成结果掩膜
    #     else:
    #         output_image = img
    #     print("图片完成检测")
    #     if is_show:
    #         cv2.imshow("seg", output_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     if is_save_folder:
    #        cv2.imwrite(imgsavepath, output_image)
    #
    # elif mode == 2:
    #     # 摄像头图像分割
    #     cap = cv2.VideoCapture(0)
    #     # 返回当前时间
    #     start_time = time.time()
    #     counter = 0
    #     while True:
    #         # 从摄像头中读取一帧图像
    #         ret, frame = cap.read()
    #         # 推理
    #         boxes, segments, _ = model(frame, conf_threshold=conf, iou_threshold=iou)
    #         # 画图
    #         if len(boxes) > 0:
    #             output_image = model.draw_and_visualize(frame, boxes, segments, vis=False, save=True,stacking=False)
    #         else:
    #             output_image = frame
    #         counter += 1  # 计算帧数
    #         # 实时显示帧数
    #         if (time.time() - start_time) != 0:
    #             cv2.putText(output_image, "FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    #             # 显示图像
    #             cv2.imshow('seg', output_image)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     # 释放资源
    #     cap.release()
    #     cv2.destroyAllWindows()
    # elif mode == 3:
    #     # 打开视频文件
    #     cap = cv2.VideoCapture(input_video_path)
    #     # 检查视频是否成功打开
    #     if not cap.isOpened():
    #         print("Error: Could not open video.")
    #         exit()
    #     # 读取视频的基本信息
    #     frame_width = int(cap.get(3))
    #     frame_height = int(cap.get(4))
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     # 定义视频编码器和创建VideoWriter对象
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 根据文件名后缀使用合适的编码器
    #     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    #     # 初始化帧数计数器和起始时间
    #     frame_count = 0
    #     start_time = time.time()
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("Info: End of video file.")
    #             break
    #         # 推理
    #         start_time_instance = time.time()
    #         boxes, segments, _ = model(frame, conf_threshold=conf, iou_threshold=iou)
    #         end_time_instance = time.time()-start_time_instance
    #         # 画图
    #         if len(boxes) > 0:
    #             output_image = model.draw_and_visualize(frame, boxes, segments, vis=False, save=True,stacking=False)
    #         else:
    #             output_image = frame
    #         # 计算并打印帧速率
    #         frame_count += 1
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time
    #         if elapsed_time > 0:
    #             fps = frame_count / elapsed_time
    #
    #             ##print(f"FPS: {fps:.2f}")
    #             # 设置文本参数
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             font_scale = 1
    #             font_color = (0, 255, 0)  # 绿色
    #             font_thickness = 2
    #             line_type = cv2.LINE_AA
    #             # 构造 FPS 字符串
    #             fps_text = f"FPS: {fps:.2f},inference_time{end_time_instance*1000:.2f}ms"
    #
    #             # 获取文本的大小
    #             (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
    #             # 设置文本位置（左上角）
    #             text_x = 10
    #             text_y = text_height + 10
    #             # 在图像上绘制文本
    #             cv2.putText(output_image, fps_text, (text_x, text_y), font, font_scale, font_color, font_thickness,
    #                         line_type)
    #             # 写入到视频文件
    #             out.write(output_image)
    #
    #         # 将处理后的帧写入输出视频
    #         out.write(output_image)
    #         #（可选）实时显示处理后的视频帧
    #         cv2.imshow("Output Video", output_image)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     # 释放资源
    #     cap.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    # else:
    #     print("输入错误，请检查mode的赋值")
