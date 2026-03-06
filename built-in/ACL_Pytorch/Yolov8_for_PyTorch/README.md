# YOLOv8 推理指导

## 概述

YOLO 系列是经典的 one-stage 目标检测算法，在工业界应用广泛。YOLOv8 作为其经典版本，在继承先前版本优点的同时，进一步提升了检测精度。本指导以 YOLOv8s 为例，说明如何将 PyTorch 框架的 YOLOv8 模型转换为昇腾离线模型（OM），并在昇腾处理器上完成推理验证。

**本模型支持的任务类型**

- 目标检测
- 图像分类

**参考实现**

- 代码仓：https://github.com/ultralytics/ultralytics
- 提交 tag：`8.4.2`
- 模型名称：yolov8

### 输入输出数据（以检测模型为例）

| 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
| -------- | -------- | ------------------------- | ------------ |
| images   | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |

| 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
| -------- | -------- | --------------------- | ------------ |
| output0  | FLOAT32  | batchsize x 84 x 8400 | ND           |

> 分类模型输入输出格式请参考官方文档。

## 推理环境准备

**表 1** 版本配套表

| 配套              | 版本      | 环境准备指导                                                                                 |
| ----------------- | --------- | -------------------------------------------------------------------------------------------- |
| 固件与驱动        | 25.2.0    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN              | 8.5.0     |                                                                                              |
| Python            | 3.11.6    |                                                                                              |
| PyTorch           | 2.1.0     |                                                                                              |

> **说明**：Atlas 300I Duo 推理卡请根据 CANN 版本选择对应的固件与驱动版本。建议前往[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)拉取mindie:2.3.0-300I-Duo-py311-openeuler24.03-lts版本镜像。

## 快速上手

### 获取源码

1. 克隆本仓库并进入相应目录：
   
   ```bash
   git clone https://gitcode.com/Ascend/modelzoo-GPL.git
   cd modelzoo-GPL/built-in/ACL_Pytorch/Yolov8_for_PyTorch
   ```
2. 安装依赖：
   
   ```bash
   pip3 install -r requirements.txt
   ```

### 准备数据集

#### 检测任务数据集（COCO2017 val）

```bash
mkdir -p datasets
cd datasets
# 下载标签及验证集图片
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
unzip coco2017labels-segments.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco/images
cd ..
```

数据集目录结构如下：

```
datasets/
└── coco/
    ├── annotations/
    │   └── instances_val2017.json
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    │       ├── 00000000139.jpg
    │       └── ...
    ├── labels/
    │   ├── train2017/
    │   └── val2017/
    │       ├── 00000000139.txt
    │       └── ...
    ├── LICENSE
    ├── README.txt
    ├── test-dev2017.txt
    ├── train2017.txt
    └── val2017.txt
```

#### 分类任务数据集（ImageNet2012 val）

ImageNet 数据集需从官网注册下载。请先访问 [ImageNet官网](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) 申请下载，获得验证集压缩包 `ILSVRC2012_img_val.tar`（约 6.3GB）。

```bash
mkdir -p datasets/imagenet
cd datasets/imagenet
# 将下载的压缩包放到当前目录，解压
tar -xvf /path/to/your/download/ILSVRC2012_img_val.tar -C ILSVRC2012_img_val
cd ILSVRC2012_img_val

# 下载并运行整理脚本，将图片按类别放入子文件夹
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh   # 执行后每张图片会被移动到对应的类别子文件夹（如 val/n01440764/）
rm -f valprep.sh

cd ../../..
```

文件目录结构：

```
datasets/imagenet/
└── ILSVRC2012_img_val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    ├── n01443537/
    └── ... (共 1000 个子文件夹)
```

### 模型推理

#### 1. 模型转换（PyTorch → ONNX → OM）

1. **获取官方权重文件**
   从 [YOLOv8 发布页面](https://docs.ultralytics.com/zh/models/yolov8/) 下载对应版本的 `.pt` 文件，例如 `yolov8s.pt`（检测）和 `yolov8s-cls.pt`（分类）。
2. **导出 ONNX 模型**
   
   ```bash
   python3 pth2onnx.py --pt=yolov8s.pt      # 检测模型
   python3 pth2onnx.py --pt=yolov8s-cls.pt  # 分类模型
   ```
   
   参数说明：
   
   - `--pt`：权重文件路径
     执行后得到 `yolov8s.onnx` 和 `yolov8s-cls.onnx`。
3. **配置环境变量**
   
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   
   > 请根据实际 CANN 安装路径修改上述脚本路径。
4. **查询芯片型号**
   
   ```bash
   npu-smi info
   ```
   
   在输出中找到 `Name` 列，例如 `310P3`，记为 `${chip_name}`。
5. **使用 ATC 工具转换为 OM 模型**
   
   ```bash
   # 检测模型
   atc --framework=5 \
       --model=yolov8s.onnx \
       --input_format=NCHW \
       --input_shape="images:${batchsize},3,640,640" \
       --precision_mode_v2=mixed_float16 \
       --output=yolov8s_bs${batchsize} \
       --log=error \
       --soc_version=Ascend${chip_name}
   
   # 分类模型
   atc --framework=5 \
       --model=yolov8s-cls.onnx \
       --input_format=NCHW \
       --input_shape="images:${batchsize},3,640,640" \
       --precision_mode_v2=mixed_float16 \
       --output=yolov8s-cls_bs${batchsize} \
       --log=error \
       --soc_version=Ascend${chip_name}
   ```
   
   **参数说明**：
   
   - `--model`：输入的 ONNX 文件
   - `--framework`：5 表示 ONNX 模型
   - `--input_format`：输入数据格式
   - `--input_shape`：模型输入形状，`${batchsize}` 需替换为实际批大小（如 1, 4, 8, 16）
   - `--precision_mode_v2`：精度模式，推荐 `mixed_float16`
   - `--output`：输出的 OM 模型文件名前缀
   - `--log`：日志级别
   - `--soc_version`：昇腾 AI 处理器型号，例如 `Ascend310P3`
   
   转换成功后生成 `yolov8s_bs${batchsize}.om` 和 `yolov8s-cls_bs${batchsize}.om`。

#### 2. 安装 ais_bench 推理工具

访问 [ais_bench 推理工具代码仓](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，根据 README 文档完成安装。

#### 3. 配置检测任务数据集文件（仅检测模型需要）

找到 ultralytics 安装路径下的 `coco.yaml`：

```bash
pip show ultralytics
# 进入 Location 指示的路径，编辑 ultralytics/cfg/datasets/coco.yaml
```

根据实际数据集路径修改以下字段：

- `path`：COCO 数据集根目录（例如 `datasets/coco`）
- `train`：训练集路径文本文件（`train2017.txt`）
- `val`：验证集路径文本文件（`val2017.txt`）
- `test`：测试集路径文本文件（`test-dev2017.txt`）

#### 4. 执行推理与精度验证

运行 `detect_infer.py`（检测）或 `classify_infer.py`（分类）脚本，对 OM 模型进行推理，结果默认保存在 `./runs/detect/train/predictions.json`，精度结果打印在终端。

```bash
# 检测任务
python3 detect_infer.py \
    --om=yolov8s_bs${batchsize}.om \
    --batch_size=${batchsize} \
    --device_id=0 \
    --imgsz=640

# 分类任务
python3 classify_infer.py \
    --om=yolov8s-cls_bs${batchsize}.om \
    --data=datasets/imagenet/ILSVRC2012_img_val \
    --batch_size=${batchsize} \
    --device_id=0 \
    --imgsz=224
```

**参数说明**：

- `--imgsz`：模型输入尺寸
- `--om`：OM 模型文件路径
- `--batch_size`：OM 模型的 batch size（需与转换时一致）
- `--device_id`：使用的昇腾设备 ID
- `--data`：分类任务的数据集根目录

#### 5. 性能验证

使用 `ais_bench` 工具的纯推理模式测试 OM 模型的吞吐性能：

```bash
# 检测模型
python3 -m ais_bench --model=yolov8s_bs8.om --loop=100

# 分类模型
python3 -m ais_bench --model=yolov8s-cls_bs8.om --loop=100
```

**参数说明**：

- `--model`：OM 模型路径
- `--loop`：循环推理次数，用于计算平均性能

## 模型推理性能与精度

调用 ACL 接口推理，在昇腾 310P3 上测试得到的性能与精度参考数据如下。

### 检测模型（COCO val2017）

| 模型     | 最优 Batch | 数据集       | 精度阈值 (conf=0.001, iou=0.7) | mAP@0.5 | 性能 (fps) |
| -------- | ---------- | ------------ | ------------------------------- | ------- | ---------- |
| yolov8n  | 8          | coco val2017 | conf=0.001 iou=0.7              | 52.5    | 753.23     |
| yolov8s  | 8          | coco val2017 | conf=0.001 iou=0.7              | 61.6    | 443.60     |
| yolov8m  | 8          | coco val2017 | conf=0.001 iou=0.7              | 67.0    | 214.19     |
| yolov8l  | 8          | coco val2017 | conf=0.001 iou=0.7              | 69.6    | 148.60     |
| yolov8x  | 8          | coco val2017 | conf=0.001 iou=0.7              | 70.6    | 100.32     |

### 分类模型（ImageNet2012 val）

| 模型        | 最优 Batch | 数据集           | Top-1 准确率 | Top-5 准确率 | 性能 (fps) |
| ----------- | ---------- | ---------------- | ------------ | ------------ | ---------- |
| yolov8s-cls | 8          | ImageNet2012 val | 73.7%        | 91.7%        |5204     |

> **说明**：性能数据为纯推理吞吐量（fps），实际性能可能因硬件环境略有差异。
