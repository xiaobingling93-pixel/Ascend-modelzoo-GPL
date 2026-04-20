# YOLOV26-推理指导

- [YOLOV26 推理指导](#yolov26-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [获取数据集](#获取数据集)
  - [获取权重](#获取权重)
  - [执行推理](#执行推理)
  - [性能精度数据](#性能精度数据)

******

# 概述
YOLO 系列是经典的 one-stage 目标检测算法，在工业界应用广泛。YOLOv26 作为其经典版本，在继承先前版本优点的同时，进一步提升了检测精度。本指导以 YOLOV26 为例，说明如何将 PyTorch 框架的 YOLOv26 模型转换为昇腾离线模型（OM），并在昇腾处理器上完成推理验证。

| 本模型支持的任务类型 |
|------------------ |
| 目标检测          | 

代码仓：https://github.com/ultralytics/ultralytics

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1**  版本配套表
  
  | 配套                                                      | 版本          | 环境准备指导                                                                                        |
  | ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                               | 25.5   | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.5.0       | -                                                                                             |
  | Python                                                  | 3.11.14        | -                                                                                             |
  | PyTorch                                                 | 2.1.0      | -                                                                                             |
  | Ascend Extension PyTorch                                | 2.1.0.post17 | -                                                                                             |
  | 说明：Atlas 800T A2/Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |

# 快速上手

## 获取源码


1. 获取ModelZoo-GPL代码  

   ```
   git clone https://gitcode.com/Ascend/modelzoo-GPL.git
   cd modelzoo-GPL/built-in/ACL_Pytorch/Yolov26_for_PyTorch
   ``` 
2. 安装依赖  
   
   ```
   pip3 install -r ../requirements.txt
   ```

## 获取数据集

检测任务数据集（COCO2017 val）

```bash
mkdir -p datasets
cd datasets
# 下载标签及验证集图片
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
unzip coco2017labels-segments.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco/
cd ..
```
最后数据集按照如下的目录结构组织：
```
datasets/
└── coco/
    ├── annotations/
    │   └── instances_val2017.json
    │   val2017/
    │   ├── 00000000139.jpg
    │   └── ...
```
## 获取权重

选择下载[模型权重](https://www.modelscope.cn/models/Ultralytics/YOLO26/files)，

下面步骤以yolo26n.pt为例，下载权重文件[yolov26n.pt](https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/9b/09/cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef?filename=yolo26n.pt&namespace=Ultralytics&repository=YOLO26&revision=master&tag=model&auth_key=1775635526-3ecfa7f8fe824ae98921b43405c747e5-0-a6d041fa3cedff3b3f5d505d53a1b6d8)

## 执行推理

### 1.导出onnx 模型

```
python  pth2onnx.py  --pt  yolo26n.pt
```

参数说明：

--pt：权重文件路径 执行后得到 yolo26n.onnx 

###  2.配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
### 3.执行atc模型转化

执行如下命令，获取芯片类型
```
npu-smi info
```
执行atc命令，将yolo26n.onnx转换为yolo26n.om模型
```
batchsize=16
chip_name=xxx

atc --framework=5 \
      --model=yolo26n.onnx \
      --input_format=NCHW \
      --input_shape="images:${batchsize},3,640,640" \
      --precision_mode_v2=mixed_float16 \
      --output=yolov26  \
      --log=error \
      --soc_version=Ascend${chip_name}
```
备注: 其中chip_name需根据实际芯片类型填写，通过npu-smi info查询

   **参数说明**：
   
   - `--model`：输入的 ONNX 文件
   - `--framework`：5 表示 ONNX 模型
   - `--input_format`：输入数据格式
   - `--input_shape`：模型输入形状，`${batchsize}` 需替换为实际批大小（如 1, 4, 8, 16）
   - `--precision_mode_v2`：精度模式，推荐 `mixed_float16`
   - `--output`：输出的 OM 模型文件名前缀
   - `--log`：日志级别
   - `--soc_version`：昇腾 AI 处理器型号

### 4.推理精度测试

首先安装 ais_bench 推理工具，访问 [ais_bench 推理工具代码仓](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，根据 README 文档完成安装。

找到 ultralytics 安装路径下的 `coco.yaml`：

```bash
pip show ultralytics
# 进入 Location 指示的路径，编辑 ultralytics/cfg/datasets/coco.yaml
```
根据实际数据集路径修改以下字段：

- `path`：COCO 数据集根目录（例如 `datasets/coco`）
- `train`：训练集路径文本文件，可以不用修改（`train2017.txt`）
- `val`：验证集路径文本文件（例如`val2017`）
- `test`：测试集路径文本文件，可以不用修改（`test-dev2017.txt`）

```bash
python3 detect_infer.py \
    --om=yolo26n.om \
    --batch_size=${batchsize} \
    --device_id=0 \
    --imgsz=640
```
**参数说明**：

- `--imgsz`：模型输入尺寸
- `--om`：OM 模型文件路径
- `--batch_size`：OM 模型的 batch size（需与转换时一致）
- `--device_id`：使用的昇腾设备 ID
- `--data`：分类任务的数据集根目录

### 5. 推理性能测试

使用 `ais_bench` 工具的纯推理模式测试 OM 模型的吞吐性能：

```bash
python3 -m ais_bench --model=yolo26n.om --loop=100
```
**参数说明**：

- `--model`：OM 模型路径
- `--loop`：循环推理次数，用于计算平均性能

## 模型推理性能与精度结果

调用 ACL 接口推理，在昇腾 Atlas 800T A2/Atlas 300I DUO 上测试得到的bs=8,图片尺寸 640x640 场景下的性能与精度（COCO val2017）参考数据如下。

| 模型     | 硬件平台 | 数据集       |  NPU 精度mAP |  官方精度mAP |性能 (fps) |
| -------- | ---------- | ------------ | ------- |------- | ---------- |
| yolo26n  | 800T A2          | coco val2017           | 40.2    |40.1    | 1545.78     |
| yolo26n  | 300I DUO          | coco val2017           | 40.2    |40.1    | 461.13     |
