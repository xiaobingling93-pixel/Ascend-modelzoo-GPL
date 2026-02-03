# YOLOV13(TorchAir)-推理指导

- [YOLOV13(TorchAir)-推理指导](#yolov13torchair-推理指导)
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

| 本模型支持的任务类型 |
|------------------ |
| 目标检测          | 

- 版本说明：
  
  ```
  url=https://github.com/iMoonLab/yolov13.git
  commit_id=73289949533efac82bb5f72ec19b746618656bd2
  model_name=YOLOV13
  ```

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1**  版本配套表
  
  | 配套                                                      | 版本          | 环境准备指导                                                                                        |
  | ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                               | 25.3.rc1   | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.3.RC2       | -                                                                                             |
  | Python                                                  | 3.11.10        | -                                                                                             |
  | PyTorch                                                 | 2.1.0      | -                                                                                             |
  | Ascend Extension PyTorch                                | 2.1.0.post10 | -                                                                                             |
  | 说明：Atlas 800I A2/Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |

# 快速上手

## 获取源码


1. 获取ModelZoo-GPL代码  

   ```
   git clone https://gitcode.com/Ascend/modelzoo-GPL.git
   cd modelzoo-GPL/built-in/ACL_Pytorch/Yolov13_for_PyTorch
   ``` 

2. 获取`Pytorch`源码  
   
   ```
   git clone https://github.com/iMoonLab/yolov13.git
   cd yolov13
   git reset --hard 73289949533efac82bb5f72ec19b746618656bd2
   git apply ../diff.patch
   ```

3. 安装依赖  
   
   ```
   pip3 install -r ../requirements.txt
   ```

## 获取数据集

1. 在yolov13目录下，新建dataset文件夹，下载COCO-2017数据集的[图片](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fzips%2Fval2017.zip)与[标注](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fannotations%2Fannotations_trainval2017.zip)，放置dataset目录下
   
   ```
   yolov13
   └── dataset
      └── annotations
      └── val2017
   ```

2. 利用dataset_convert.py脚本，将COCO数据集转化为YOLO仓支持的格式
   
   ```bash
   python3 ../dataset_convert.py --data_path=dataset/annotations/
   ```
  - 参数说明
    - data_path: coco数据集annotations目录，须保证annotations目录下只有instances_val2017.json

    执行后生成coco_converted文件夹

3. 移动image文件夹到coco_converted内，并将yolov13/ultralytics/cfg/datasets/coco.yaml的line12将path改coco_converted目录的绝对路径，并将val的值修改为images/val2017
   
   ```
   mv dataset/val2017 coco_converted/images/
   ```
   
   完成后数据集结构如下
   
   ```
   yolov13
   └── coco_converted
      ├── images
         └── val2017
            └── ***.jpg
      └── labels
         └── val2017
            └── ***.txt
   ```

## 获取权重

选择下载[模型权重](https://github.com/iMoonLab/yolov13?tab=readme-ov-file#2-validation)，下面步骤以yolov13l.pt为例，下载权重文件[yolov13l.pt](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt)，放到yolov13目录下

```
wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt
```

## 执行推理

配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

指定使用NPU ID，默认为0

```
export ASCEND_RT_VISIBLE_DEVICES=0
```

yolov13目录下运行推理脚本infer.py

```
mv ../infer.py .
python3 infer.py --pth=yolov13l.pt --dataset=coco.yaml --batchsize=16
```

- 参数说明
  - pth: 模型权重
  - dataset: 数据集信息文件，一般在ultralytics/cfg/datasets/中
  - batchsize: 数据集推理batchsize

推理执行完成后，数据集精度和数据集性能会执行打屏

## 性能精度数据

以bs16在COCO-2017数据集上的推理为例
|模型    |芯片     |性能(per image)  |精度(mAP50) |GPU精度(mAP50) |
|------- |--------|-----------------|------------|--------------|
|yolo13n |300I DUO|8.6ms            |0.573       |0.573         |
|yolo13s |300I DUO|10.5ms           |0.647       |0.646         |
|yolo13l |300I DUO|23.5ms           |0.704       |0.704         |
|yolo13x |300I DUO|34.1ms           |0.716       |0.716         |

注：该模型支持动态shape数据集推理，输入shape长边会处理为640，短边以原尺寸scale
