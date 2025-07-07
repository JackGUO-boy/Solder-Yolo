## 焊点数据集：
通过网盘分享的文件：Train_NG_OK.zip
链接: https://pan.baidu.com/s/1d4qX475AMOghE_-HrxFM4w?pwd=6666 提取码: 6666

数据格式为yolo实例分割格式

## 标签
Solder_joint
Null

其中Solder_joint表示焊点，Null表示非焊点，即露铜部分

## 学生模型
在ultralytics/cfg/models/v8/student2_Attention.yaml

## 蒸馏训练
运行distill.py即可；在distill.py中设置好学生模型、教师模型、数据集、优化器、蒸馏损失计算方式（CWD、mgd、logit等）

## 配置文件
在ultralytics/cfg/default.yaml中，最后添加了有关蒸馏的一些配置

## 预训练权重
在model_data文件夹下面

## 其他
tools文件夹下面有模型推理程序。需要先把yolo训练的模型或蒸馏的模型导出为onnx，再将模型路径填写到对应为止。程序采用pyqt编写、

--------------------------------------------------------------------------------------------------------------
## Solder Joint Dataset
The dataset is shared via Baidu Netdisk: Train_NG_OK.zip
Link: https://pan.baidu.com/s/1d4qX475AMOghE_-HrxFM4w?pwd=6666
Extraction Code: 6666

Format: YOLO instance segmentation format

## Labels:

Solder_joint: Represents solder joints
Null: Represents non-solder areas (exposed copper)

## Student Model
Location: ultralytics/cfg/models/v8/student2_Attention.yaml

## Knowledge Distillation Training
Run distill.py to start training.

Configure the following in distill.py:
  Student model architecture
  Teacher model weights
  Dataset path
  Optimizer settings
  Distillation loss function (CWD, mgd, logit, etc.)
## Configuration File
Additional distillation-related configurations are added to the end of:
ultralytics/cfg/default.yaml

## Pre-trained Weights
Stored in the model_data/ directory.

## Inference Tools
Export the YOLO-trained or distilled model to ONNX format.
Update the model path in the inference script.
Run the inference GUI (PyQt-based) located in:
tools/ directory.
