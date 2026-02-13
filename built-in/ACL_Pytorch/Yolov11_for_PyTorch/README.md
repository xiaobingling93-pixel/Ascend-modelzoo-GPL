# YOLOV11-推理指导

- [YOLOV11-推理指导](#yolov11-推理指导)
- [概述](#概述)
  - [YOLO11](#yolo11)
  - [TorchAir方案（推荐）](#torchair方案推荐)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [获取数据集](#获取数据集)
  - [获取权重](#获取权重)
  - [执行推理](#执行推理)
    - [补充说明](#补充说明)
  - [推理结果](#推理结果)
  - [补充方案：OM方式推理](#补充方案om方式推理)
    - [准备工作](#准备工作)
    - [执行步骤](#执行步骤)
      - [参数及变量含义](#参数及变量含义)
    - [OM方式推理结果](#om方式推理结果)

******

# 概述

| 本模型支持的任务类型 |
|------------------- |
| 目标检测/detect     | 
| 分割/segment        |
| 分类/classify       |
| 姿态估计/pose       |
| 定向/obb            |

## YOLO11
YOLO（You Only Look Once）是一种流行的物体检测和图像分割模型，由华盛顿大学的 Joseph Redmon 和 Ali Farhadi 开发。YOLO 于 2015 年推出，因其高速和高精度而广受欢迎。
最新的YOLO11 模型可在多项任务中提供最先进的 (SOTA) 性能，包括物体检测、分割、姿势估计 、跟踪和分类，可在各种人工智能应用和领域中部署。

## TorchAir方案（推荐）
TorchAir（Torch Ascend Intermediate Representation）是昇腾Ascend Extension for PyTorch（torch_npu）的图模式能力扩展库，提供了昇腾设备亲和的torch.compile图模式后端，实现了PyTorch网络在昇腾NPU上的图模式推理加速以及性能优化。

310P RC设备暂时不支持TorchAir方案。

- 版本说明：
  
  ```
  url=https://github.com/ultralytics/ultralytics.git
  commit_id=e74b035
  model_name=YOLOV11
  ```

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1**  版本配套表
  
  | 配套                                                      | 版本          | 环境准备指导                                                                                        |
  | ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                   | 24.0.RC3    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.3.RC1       | -                                                                                             |
  | Python                                                  | 3.11        | -                                                                                             |
  | PyTorch                                                 | 2.1.0       | -                                                                                             |
  | Ascend Extension PyTorch                                | 2.1.0.post10 | -                                                                                             |
  | 说明：Atlas 800I A2/Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |

# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   
   ```
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   git reset --hard e74b035
   git apply ../diff.patch
   cd ..
   export PYTHONPATH=./ultralytics:$PYTHONPATH
   ```

2. 安装依赖  
   
   ```
   pip3 install -r requirements.txt
   ```

## 获取数据集
本项目运行会自动下载数据集，无需手动下载，建议采用自动下载方案。
如自动下载失败，可选择手动下载数据集，并放置在`./datasets`目录下。手动下载方式详见[安装指南](../docs/dataset_preparation.md)


## 获取权重

本项目运行会自动下载权重文件，无需手动下载，建议采用自动下载方案。
如自动下载失败，可选择手动下载权重，并放置在运行目录下，默认为`${YOLOV11_PATH}`目录下。手动下载地址如下：

- detect 任务：
下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
  
- segment 任务
下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt


- classify 任务
下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt

- pose 任务：
下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt


- obb 任务：
下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt



## 执行推理
```bash
python3 infer.py --task=${task_name} --model_path=${pth} --dataset=${dataset} --batchsize=${batch_size} --device=${device_id}
```

### 补充说明
- 参数含义
  - model_path: 模型路径
  - task：任务类型
  - dataset： 选用的数据集
  - device：推理使用的芯片id
- 变量含义
  - ${pth}:  pt结尾的模型，如 yolo11n.pt，可缺省
  - ${task_name}: 任务名，可选择范围为：[detect, segment, classify, pose, obb]
  - ${dataset}: 数据集配置或路径，可选择范围[coco.yaml, coco-pose.yaml, DOTAv1.yaml, path/to/imagenet], 其中：
    - coco.yaml 对应 detect 以及 segment 任务
    - coco-pose.yaml 对应 pose 任务
    - DOTAv1.yaml 对应 obb 任务
    - path/to/imagenet为此数据集实际路径，对应 classify任务，此数据集需要提前手动下载。
  - ${onnx_path}: 导出的onnx路径，因为默认在当前路径，因此一般形如 yolo11n_bs4.onnx
  - ${chip_name}: 芯片名，可通过命令行执行`npu-smi info`查看
  - ${device_id}: 设备id，必须为数字，如0、1、2等
  - ${batch_size}: 推理的batchsize,如1、4、8等

推理执行完成后，数据集精度和数据集性能会执行打屏

## 推理结果

如下展示了不同芯片上不同任务在对应数据集上的精度与性能数据

**800I A2 芯片 TorchAIR 方式推理结果**
|任务类型            |bs         | 性能(per image) |精度(mAP)| 精度结果|
|----------------|-----------|-----------------|---------|---------|
|detect(yolo11l.pt) |16         |1.7ms            |val-mAP50-95     | 53.3  |


**300I DUO 芯片 TorchAIR 方式推理结果**
|任务类型            |bs         | 性能(per image) |精度(mAP)| 精度结果|
|----------------|-----------|-----------------|---------|---------|
|detect      |1          |6.3ms            |val-mAP50-95     | 39.2  |
|      |4          |2.5ms            |val-mAP50-95     | 39.2  |
|      |8          |2.1ms            |val-mAP50-95     | 39.2  |
|segment  |1          |9.4ms            |mAP50-95     | 38.9  |
|  |4          |3.1ms            |mAP50-95     | 38.9  |
|  |8          |2.5ms            |mAP50-95     | 38.9  |
|classify  |1          |2.9ms            |acc-top5     | 89.4 |
|  |4          |0.7ms            |acc-top5      | 89.4   |
|  |8          |0.4ms            |acc-top5     | 89.4   |
|pose  |1          |12.4ms            |POSE-mAP50     | 81.0  |
|  |4          |6.9ms            |POSE-mAP50     | 81.0  |
|  |8          |6.5ms            |POSE-mAP50     | 81.0  |
|obb  |1          |17.7ms            |mAP50-95     | 47.1  |
|  |4          |13.8ms            |mAP50-95     | 47.1  |
|  |8          |13.7ms            |mAP50-95     | 47.1  |

## 补充方案：OM方式推理

昇腾张量编译器（Ascend Tensor Compiler，简称ATC）是异构计算架构CANN体系下的模型转换工具，它可以将开源框架的网络模型以及Ascend IR定义的单算子描述文件（JSON格式）转换为昇腾AI处理器支持的.om格式离线模型。om离线模型采用ais_bench工具推理能够大幅提高纯推理阶段性能。

### 准备工作

获取源码、安装依赖、权重下载、数据集准备工作同前述TorchAir路线。

### 执行步骤

```bash
# 权重文件转onnx格式，执行这一步需要提前手动下载权重文件或至少执行过一次TorchAir方式的推理
python3 pth2onnx.py --pt=${pt} --batch=${batch_size} --simplify=True


# 上一步导出的onnx文件转om格式
bash convert_to_om.sh ${onnx_path} [${batch_size}] [${chip_name}]

# 执行推理
python3 infer.py --model_path=${om_path} --task=${task_name} --dataset=${dataset} --batchsize=${batch_size} --device=${device_id}
```

#### 参数及变量含义
与前述TorchAir相同参数及变量不再赘述
  - ${pt}: pt结尾的模型路径模型路径
  - simplify: 是否简化，此参数可缺省，默认为True。如使用RC环境，需设置为False
  - ${om_path}:为om结尾的模型，如 yolo11n_bs4.om，不可缺省。
  - ${onnx_path}: 导出的onnx路径，因为默认在当前路径，因此一般形如 yolo11n_bs4.onnx

推理执行完成后，数据集精度和数据集性能会执行打屏

### OM方式推理结果
**300I DUO 芯片 OM 方式推理结果**
|任务类型          |bs         | 性能(per image) |精度(mAP)| 精度结果|
|----------------|-----------|-----------------|---------|---------|
|detect  |1          |6.0ms            |val-mAP50-95     | 39.1  |
|  |4          |2.9ms            |val-mAP50-95     | 39.1  |
|  |8          |3.2ms            |val-mAP50-95     | 39.1  |
|segment  |1          |6.0ms            |mAP50-95     | 38.7  |
|  |4          |5.1ms            |mAP50-95     | 38.7  |
|  |8          |5.2ms            |mAP50-95     | 38.7  |
|classify  |1          |1.4ms            |acc-top5      | 89.4   |
|  |4          |0.3ms            |acc-top5     | 89.4   |
|  |8          |0.3ms            |acc-top5      | 89.4   |
|pose  |1          |5.1ms            |POSE-mAP50     | 81.0  |
|  |4          |2.6ms            |POSE-mAP50     | 81.0  |
|  |8          |2.8ms            |POSE-mAP50     | 81.0  |
|obb  |1          |8.4ms            |mAP50-95     | 47.1  |
|  |4          |6.8ms            |mAP50-95     | 47.1  |
|  |8          |7.0ms            |mAP50-95     | 47.1  |

**310P RC 设备 OM 方式推理结果**
|任务类型          |bs         | 性能(per image) |精度(mAP)| 精度结果|
|----------------|-----------|-----------------|---------|---------|
|detect  |1          |8.7ms            |val-mAP50-95     | 39.1  |
|  |8          |5.1ms            |val-mAP50-95     | 39.1  |
|classify  |1          |4.3ms            |acc-top5      | 89.4   |
|  |8          |0.7ms            |acc-top5      | 89.4   |
|pose  |1          |15.9ms            |POSE-mAP50     | 79.3  |
|  |8          |4.5ms            |POSE-mAP50     | 79.3  |
|obb  |1          |16.3ms            |mAP50-95     | 47.1  |
|  |8          |9.4ms            |mAP50-95     | 47.1  |
