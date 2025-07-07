# Ultralytics YOLO 🚀, AGPL-3.0 license
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from PIL import Image, ImageDraw, ImageFont
import os
class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        #初始化损失向量：
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        #解析预测结果：特征图、预测掩码系数、原型掩码
        # feats 是一个列表 ，包含三个尺度的特征图 shape=torch.Size([2, 66, 80, 80]) torch.Size([2, 66, 160, 160]) torch.Size([2, 66, 40, 40])
        # pred—_mask shape = torch.Size([2, 32, 33600])
        # proto shape = torch.Size([2, 32, 160, 160])
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        # print("1111111111")
        # print(feats[0].shape,feats[1].shape,feats[2].shape)
        # print("222222222")
        # print(pred_masks.shape)
        # print("333333333")
        # print(proto.shape)

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        #将 feats 列表中的特征张量连接在一起，并拆分为两个部分
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        #调整预测结果的形状：
        # B, grids, ..
        #调整 pred_scores 张量的维度顺序，并确保张量在内存中是连续的
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        #调整 pred_distri 张量的维度顺序，并确保张量在内存中是连续的。
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        #调整 pred_masks 张量的维度顺序，并确保张量在内存中是连续的。
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()


        #获取 pred_scores 张量的数据类型。
        dtype = pred_scores.dtype
        # 计算图像尺寸：
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        #生成锚点：生成的 anchor_points 是锚点的位置，stride_tensor 是步幅张量。
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            # 获取批次索引并调整其形状。
            batch_idx = batch["batch_idx"].view(-1, 1)
            #将批次索引、类别标签和边界框拼接在一起，形成目标数据。
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            #预处理目标数据。主要是进行归一化和调整尺度
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            #将目标数据拆分为类别标签和边界框。
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            # 生成目标掩码。.gt_(0) 判断边界框和是否大于 0，生成二值掩码，指示哪些边界框是有效的。
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        #解码预测边界框：返回的 pred_bboxes 是解码后的边界框，形状为 [batch_size, num_anchors, 4]，
        # 其中 num_anchors 是锚点的数量，每个边界框包含 4 个坐标（xyxy 格式）。
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        #计算目标边界框和得分，确定前景掩码和目标索引。
        # 返回值：目标边界框 (target_bboxes)、目标得分 (target_scores)、前景掩码 (fg_mask)、
        # 目标索引 (target_gt_idx)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        #计算目标得分的总和，确保最小值为 1。
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        #计算分类损失。
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

           #边界框损失：
        if fg_mask.sum():#检查是否存在前景掩码。
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,#是预测的边界框分布。
                pred_bboxes,#是解码后的预测边界框。
                anchor_points,#是锚点位置。
                target_bboxes / stride_tensor,#是归一化后的目标边界框。
                target_scores,# 是目标得分。
                target_scores_sum,#是目标得分的总和。
                fg_mask,# 是前景掩码。
            )

            #将批次中的掩码转换为浮点数，并移动到指定的设备上（例如 GPU）。
            masks = batch["masks"].to(self.device).float()
            #如果掩码的形状与预测的原型掩码形状不匹配，则进行下采样以匹配形状。
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            #计算分割损失。
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,# 前景掩码。
                masks, #处理后的真实掩码。
                target_gt_idx,#每个锚点对应的真实目标索引。
                target_bboxes,#每个锚点对应的真实边界框。
                batch_idx,# 批次索引。
                proto, # 原型掩码。
                pred_masks,#预测的掩码系数。
                imgsz,#图像大小。
                self.overlap#掩码是否重叠。
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        #应用不同损失项的权重增益（gain），得到最终的损失值。
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        # # 生成最终的预测掩码
        # #predicted_masks = self.get_predicted_masks(pred_masks, proto,fg_mask,target_gt_idx,batch_idx,imgsz)
        # self.save_predicted_masks(pred_masks, proto,fg_mask,target_gt_idx,batch_idx,imgsz,save_path='D:/yolov8/ultralytics_seg_dill/runs/seg_png',pred_scores=pred_scores)

        # loss(box, cls, dfl)
        return loss.sum() * batch_size, loss.detach()

    def nms(self,boxes, scores, iou_threshold):
        """
        非极大值抑制
        """
        keep = []
        _, idxs = scores.sort(descending=True)

        while idxs.numel() > 0:
            i = idxs[0]
            keep.append(i)

            if idxs.numel() == 1:
                break

            iou = self.box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])
            idxs = idxs[1:][iou.squeeze() <= iou_threshold]

        return keep

    def box_iou(self, box1, box2):
        """
        计算两个边界框之间的IoU
        """
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        area1 = (box1[:, 2:] - box1[:, :2]).prod(1)
        area2 = (box2[:, 2:] - box2[:, :2]).prod(1)
        return inter / (area1[:, None] + area2 - inter)

    def masks_to_boxes(self, masks):
        """
        将掩码转换为边界框
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)
        n, h, w = masks.shape
        boxes = torch.zeros((n, 4), device=masks.device)
        for i in range(n):
            y, x = torch.where(masks[i])
            if y.numel() == 0 or x.numel() == 0:
                continue
            boxes[i, 0] = x.min().float()
            boxes[i, 1] = y.min().float()
            boxes[i, 2] = x.max().float()
            boxes[i, 3] = y.max().float()
        return boxes

    def save_predicted_masks(self, pred_masks, proto, fg_mask, target_gt_idx, batch_idx, img_shape, pred_scores,
                             upsample=False,
                             save_path='output', confidence_threshold=0.3):
        """
        从模型输出生成预测的掩码，并将每个锚点的图像分开保存在一张大的白色图像上。

        参数:
            pred_masks (torch.Tensor): 预测的掩码系数，形状为 [batch_size, num_anchors, 32]。
            proto (torch.Tensor): 原型掩码，形状为 [batch_size, 32, H, W]。
            fg_mask (torch.Tensor): 前景掩码，形状为 [batch_size, num_anchors]。
            target_gt_idx (torch.Tensor): 每个锚框的真实目标索引，形状为 [batch_size, num_anchors]。
            batch_idx (torch.Tensor): 批次索引，形状为 [num_labels_in_batch, 1]。
            img_shape (tuple): 输入图像的形状 (height, width)。
            upsample (bool): 是否上采样掩码到原始图像大小。
            save_path (str): 保存分割图的路径。
            confidence_threshold (float): 置信度阈值，用于过滤锚框。
        """
        device = pred_masks.device  # 获取当前设备
        batch_size, num_anchors, _ = pred_masks.shape
        _, c, mask_h, mask_w = proto.shape
        img_h, img_w = img_shape

        os.makedirs(save_path, exist_ok=True)  # 创建保存目录

        for i in range(batch_size):
            proto_i = proto[i].view(c, -1).to(device)  # [32, H*W]
            pred_masks_i = pred_masks[i].to(device)  # [num_anchors, 32]
            fg_mask_i = fg_mask[i].to(device)  # [num_anchors]
            target_gt_idx_i = target_gt_idx[i].to(device)  # [num_anchors]

            # 仅保留前景对象的掩码
            if fg_mask_i.any():
                pred_masks_fg = pred_masks_i[fg_mask_i]  # [num_objects, 32]
                mask_idx = target_gt_idx_i[fg_mask_i]  # [num_objects]

                # 为前景对象生成预测的掩码
                masks_fg = torch.einsum('nc,ch->nh', pred_masks_fg, proto_i).sigmoid().view(-1, mask_h, mask_w).to(
                    device)  # [num_objects, H, W]

                if upsample:
                    masks_fg = F.interpolate(masks_fg.unsqueeze(0), (img_h, img_w), mode='bilinear',
                                             align_corners=False).squeeze(0).to(device)  # [num_objects, img_h, img_w]

                # 使用置信度阈值过滤锚框
                mask_scores = pred_scores[i, fg_mask[i]].sigmoid().max(1).values  # 获取最高置信度分数
                keep = mask_scores > confidence_threshold
                masks_fg = masks_fg[keep]

                # 将掩码转换为边界框并应用NMS
                boxes = self.masks_to_boxes(masks_fg)
                scores = masks_fg.view(masks_fg.shape[0], -1).sum(1)  # 使用掩码面积作为分数
                keep = self.nms(boxes, scores, iou_threshold=0.2)
                masks_fg = masks_fg[keep]

                # 创建一个大的白色图像
                num_objects = masks_fg.shape[0]
                num_cols = 8  # 每行显示8个图像
                num_rows = (num_objects + num_cols - 1) // num_cols  # 计算行数
                big_image_h = img_h if upsample else mask_h
                big_image_w = img_w if upsample else mask_w
                total_height = num_rows * (big_image_h + 20)  # 每个图像下方留出20像素用于标注
                total_width = min(num_objects, num_cols) * big_image_w
                big_image = np.ones((total_height, total_width), dtype=np.uint8) * 255

                for j in range(num_objects):
                    row = j // num_cols
                    col = j % num_cols
                    y1 = row * (big_image_h + 20)
                    y2 = y1 + big_image_h
                    x1 = col * big_image_w
                    x2 = x1 + big_image_w

                    mask_np = (masks_fg[j].detach().cpu().numpy() * 255).astype(np.uint8)  # 转为numpy数组
                    big_image[y1:y2, x1:x2] = mask_np

                    # 在每个图像下方添加标注
                    img_pil = Image.fromarray(big_image)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x1 + 5, y2 + 5), f"Anchor {j + 1}", fill=0)  # 标注文本
                    big_image = np.array(img_pil)

                # 保存图像
                big_image_pil = Image.fromarray(big_image)
                big_image_pil.save(f"{save_path}/batch_{i}_predicted_masks.png")
    @staticmethod
    def get_predicted_masks(pred_masks, proto, fg_mask, target_gt_idx, batch_idx, img_shape, upsample=False):
        """
        从模型输出生成预测的掩码，并仅保留对应于前景对象的掩码。

        参数:
            pred_masks (torch.Tensor): 预测的掩码系数，形状为 [batch_size, num_anchors, 32]。
            proto (torch.Tensor): 原型掩码，形状为 [batch_size, 32, H, W]。
            fg_mask (torch.Tensor): 前景掩码，形状为 [batch_size, num_anchors]。
            target_gt_idx (torch.Tensor): 每个锚框的真实目标索引，形状为 [batch_size, num_anchors]。
            batch_idx (torch.Tensor): 批次索引，形状为 [num_labels_in_batch, 1]。
            img_shape (tuple): 输入图像的形状 (height, width)。
            upsample (bool): 是否上采样掩码到原始图像大小。

        返回:
            (torch.Tensor): 预测的二值掩码张量，形状为 [batch_size, max_num_objects, H, W] 或 [batch_size, max_num_objects, img_shape[0], img_shape[1]]。
        """
        device = pred_masks.device  # 获取当前设备
        batch_size, num_anchors, _ = pred_masks.shape
        _, c, mask_h, mask_w = proto.shape
        img_h, img_w = img_shape

        predicted_masks = []
        max_num_objects = 0

        for i in range(batch_size):
            proto_i = proto[i].view(c, -1).to(device)  # [32, H*W]
            pred_masks_i = pred_masks[i].to(device)  # [num_anchors, 32]
            fg_mask_i = fg_mask[i].to(device)  # [num_anchors]
            target_gt_idx_i = target_gt_idx[i].to(device)  # [num_anchors]

            # 仅保留前景对象的掩码
            if fg_mask_i.any():
                pred_masks_fg = pred_masks_i[fg_mask_i]  # [num_objects, 32]
                mask_idx = target_gt_idx_i[fg_mask_i]  # [num_objects]

                # 为前景对象生成预测的掩码
                masks_fg = torch.einsum('nc,ch->nh', pred_masks_fg, proto_i).sigmoid().view(-1, mask_h, mask_w).to(
                    device)  # [num_objects, H, W]

                if upsample:
                    masks_fg = F.interpolate(masks_fg.unsqueeze(0), (img_h, img_w), mode='bilinear',
                                             align_corners=False).squeeze(0).to(device)  # [num_objects, img_h, img_w]

                max_num_objects = max(max_num_objects, masks_fg.shape[0])
                predicted_masks.append(masks_fg)
            else:
                empty_mask = torch.empty((0, mask_h if not upsample else img_h, mask_w if not upsample else img_w),
                                         device=device)
                predicted_masks.append(empty_mask)

        # 填充掩码以确保它们的大小一致
        for i in range(len(predicted_masks)):
            num_objects = predicted_masks[i].shape[0]
            if num_objects < max_num_objects:
                padding = (0, 0, 0, 0, 0, max_num_objects - num_objects)
                predicted_masks[i] = F.pad(predicted_masks[i], padding).to(device)

        # 堆叠批次的掩码
        predicted_masks = torch.stack(predicted_masks).to(
            device)  # [batch_size, max_num_objects, H, W] 或 [batch_size, max_num_objects, img_h, img_w]

        return predicted_masks
    # def get_predicted_masks(pred_masks, proto, fg_mask, target_gt_idx, batch_idx, img_shape, upsample=False):
    #     """
    #     Generate predicted masks from the model's output and keep only those corresponding to foreground objects.
    #
    #     Args:
    #         pred_masks (torch.Tensor): The predicted mask coefficients of shape [batch_size, num_anchors, 32].
    #         proto (torch.Tensor): The prototype masks of shape [batch_size, 32, H, W].
    #         fg_mask (torch.Tensor): Foreground mask of shape [batch_size, num_anchors].
    #         target_gt_idx (torch.Tensor): Ground truth indices for each anchor, shape [batch_size, num_anchors].
    #         batch_idx (torch.Tensor): Batch indices, shape [num_labels_in_batch, 1].
    #         img_shape (tuple): The shape of the input image (height, width).
    #         upsample (bool): Whether to upsample the mask to the original image size.
    #
    #     Returns:
    #         (torch.Tensor): The predicted binary mask tensor of shape [batch_size, num_objects, H, W] or [batch_size, num_objects, img_shape[0], img_shape[1]].
    #     """
    #     batch_size, num_anchors, _ = pred_masks.shape
    #     _, c, mask_h, mask_w = proto.shape
    #     img_h, img_w = img_shape
    #
    #     predicted_masks = []
    #
    #     for i in range(batch_size):
    #         proto_i = proto[i].view(c, -1)  # [32, H*W]
    #         pred_masks_i = pred_masks[i]  # [num_anchors, 32]
    #         fg_mask_i = fg_mask[i]  # [num_anchors]
    #         target_gt_idx_i = target_gt_idx[i]  # [num_anchors]
    #         batch_idx_i = batch_idx.view(-1) == i
    #
    #         # Only keep masks corresponding to foreground objects
    #         if fg_mask_i.any():
    #             pred_masks_fg = pred_masks_i[fg_mask_i]  # [num_objects, 32]
    #             mask_idx = target_gt_idx_i[fg_mask_i]  # [num_objects]
    #
    #             # Generate the predicted masks for foreground objects
    #             masks_fg = torch.einsum('nc,ch->nh', pred_masks_fg, proto_i).sigmoid().view(-1, mask_h,
    #                                                                                         mask_w)  # [num_objects, H, W]
    #
    #             if upsample:
    #                 masks_fg = F.interpolate(masks_fg.unsqueeze(0), (img_h, img_w), mode='bilinear',
    #                                          align_corners=False).squeeze(0)  # [num_objects, img_h, img_w]
    #
    #             predicted_masks.append(masks_fg)
    #
    #     # Stack the masks for the batch
    #     predicted_masks = torch.stack(
    #         predicted_masks)  # [batch_size, num_objects, H, W] or [batch_size, num_objects, img_h, img_w]
    #     return predicted_masks

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor,#真实掩码，形状为 [n, H, W]，其中 n 是对象的数量，H 和 W 分别是高度和宽度。
            pred: torch.Tensor,#预测的掩码系数，形状为 [n, 32]。
            proto: torch.Tensor,#原型掩码，形状为 [32, H, W]。
            xyxy: torch.Tensor,#归一化到 [0, 1] 范围的真实边界框，形状为 [n, 4]。
            area: torch.Tensor#每个真实边界框的面积，形状为 [n]。
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        # pred_mask.shape =  torch.Size([20, 160, 160])
        # proto.shape = torch.Size([32, 160, 160]),
        # pred.shape = torch.Size([20, 32]) ,torch.Size([70, 32]),torch.Size([50, 32])


        # gt_mask.shape =  torch.Size([90, 160, 160])
        #使用爱因斯坦求和约定将预测的掩码系数与原型掩码相乘。
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        # "in,nhw->ihw"：表示将形状为 (n, 32) 的 pred 与形状为 (32, H, W) 的 proto 相乘，
        # 得到形状为 (n, H, W) 的 pred_mask

        # print("===pred.shape")
        # print(pred.shape)
        # print("===proto.shape")
        # print(proto.shape)
        # 计算二元交叉熵损失。
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        # 对损失进行裁剪、归一化并返回总损失。
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,#一个二进制张量，形状为 [BS, N_anchors]，表示哪些锚点是正样本。
        masks: torch.Tensor,#真实掩码，形状为 [BS, H, W]（如果 overlap 为 False）或 [BS, ?, H, W]（如果 overlap 为 True）。
        target_gt_idx: torch.Tensor,#每个锚点对应的真实目标索引，形状为 [BS, N_anchors]。
        target_bboxes: torch.Tensor,#每个锚点对应的真实边界框，形状为 [BS, N_anchors, 4]。
        batch_idx: torch.Tensor,#批次索引，形状为 [N_labels_in_batch, 1]。
        proto: torch.Tensor,#原型掩码，形状为 [BS, 32, H, W]。
        pred_masks: torch.Tensor,#每个锚点的预测掩码，形状为 [BS, N_anchors, 32]。
        imgsz: torch.Tensor,#输入图像的大小，形状为 [2]，即 [H, W]。
        overlap: bool,#掩码是否重叠。
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        #获取原型掩码的高度和宽度。
        _, _, mask_h, mask_w = proto.shape
        #初始化损失变量。
        loss = 0
        #将目标边界框归一化到 [0, 1] 的范围。
        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        # 计算目标边界框的面积
        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # 将目标边界框归一化到掩码大小。
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        #计算每个批次的损失：
        # 遍历批次中的每个元素，将相关参数组合成一个元组 single_i。
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            #解包组合的元组。
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            # 如果存在前景掩码。
            if fg_mask_i.any():
                #获取前景掩码对应的目标索引。
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:#如果掩码重叠。
                    #根据目标索引生成真实掩码。
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    #将真实掩码转换为浮点数。
                    gt_mask = gt_mask.float()
                else:# 如果掩码不重叠。
                    # 根据批次索引和目标索引获取真实掩码。
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()




class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
