# YOLOV12 (TorchAir) - 推理指导

- [YOLOV12(TorchAir)-推理指导](#yolov12torchair-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [获取数据集](#获取数据集)
  - [获取权重](#获取权重)
  - [执行推理](#执行推理)
  - [精度性能数据](#精度性能数据)

---

# 概述

- 版本说明：
  
  ```
  url=https://github.com/sunsmarterjie/yolov12.git
  commit_id=51901136772609c36df65cec1131d54b4f1a44df
  model_name=YOLOV12
  ```

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1** 版本配套表
  
  | 配套                                                      | 版本            | 环境准备指导                                                                                        |
  | ------------------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                   | 25.0.rc1.b010 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.1.0         | -                                                                                             |
  | Python                                                  | 3.11          | -                                                                                             |
  | PyTorch                                                 | 2.4.0         | -                                                                                             |
  | Ascend Extension PyTorch                                | 2.4.0         | -                                                                                             |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \             | \                                                                                             |

# 快速上手

## 获取源码

1. 获取`Pytorch`源码
   
   ```
   git clone https://github.com/sunsmarterjie/yolov12.git
   cd yolov12
   git reset --hard 51901136772609c36df65cec1131d54b4f1a44df
   ```

2. 下载modelzoo中的文件并移动到yolov12目录下，安装依赖
   
   ```
   pip3 install -r infer_requirements.txt
   ```

## 获取数据集

1. 新建data文件夹，下载[COCO-2017数据集](http://images.cocodataset.org/zips/val2017.zip)与[标注](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)，放置data目录下
   
   ```
   yolov12
   └── data
      └── annotations
      └── val2017
   ```

2. 利用dataset.py脚本，将数据集内的annotation.json文件转化为YOLO支持的格式，即每张图片对应一个包含annotation信息的txt文件
   
   ```bash
   python3 dataset_convert.py --data_path=./data/annotations/
   ```
- 参数说明
  - data_path: coco数据集annotations目录，须保证目录下只有instances_val2017.json文件

执行后生成coco_converted文件夹

3. 移动image文件夹到coco_converted内，并将yolov12/ultralytics/cfg/datasets/coco.yaml的line12将path改coco_converted目录的绝对路径，并将val的值修改为images/val2017
   
   ```
   mv data/val2017 coco_converted/images/
   ```
   
   完成后数据集结构如下
   
   ```
   yolov12
   └── coco_converted
     ├── images
           └── val2017
               └── ***.jpg
     └── labels
         └── val2017
                 └── ***.txt
   ```

## 获取权重

下载权重文件[yolo12l.pt](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt)，放到yolov12目录下

```
wget https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt
```

## 执行推理

配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

修改 ultralytics/engine/validator.py 部分源码
```
patch -p2 < ../diff.patch
```

运行推理脚本infer.py

```
python3 infer.py --pth=yolov12l.pt --dataset=ultralytics/cfg/datasets/coco.yaml --batchsize=16
```

- 参数说明
  - pth: 模型权重，以yolo12l.pt为例
  - dataset: 数据集信息，一般位于ultralytics/cfg/datasets/coco.yaml
  - batchsize: 数据集推理batchsize

推理执行完成后，数据集精度会执行打屏

## 精度性能数据

以bs16在数据集上的推理为例
|模型|芯片|性能(per image)|精度(mAP)|
|------|------|------|------|
|yolo12l|800I A2|5.3ms|54.0|

注：该模型支持动态shape数据集推理，输入shape长边会处理为640，短边以原尺寸scale

