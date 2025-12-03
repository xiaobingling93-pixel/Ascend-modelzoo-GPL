# Yolov8_for_PyTorch

- [Yolov8\_for\_PyTorch](#Yolov8_for_PyTorch)
- [概述](#概述)
  - [简述](#简述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [开始训练](#开始训练)
  - [训练模型](#训练模型)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
- [公网地址说明](#公网地址说明)

# 概述

## 简述
YOLO是一个经典的物体检测网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。YOLOv8 由 Ultralytics 在 2023 年发布。YOLOv8 引入了新特性和改进，以增强性能、灵活性和效率，支持全方位的视觉人工智能任务。

- 参考实现：

  ```
  url=https://github.com/ultralytics/ultralytics.git
  commit_id=83404afff1536e0ce08c9926dbb1a1217c411914
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/modelzoo-GPL
  code_path=built-in/PyTorch/Official/cv/object_detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version | 三方库依赖版本                                 |
  |:-------------:| :----------------------------------------------------------: |
  |  PyTorch 2.1  | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 下载权重

请自行下载yolov8n-obb.pt模型权重，并保存在源码包根目录下。

## 准备数据集

1. 在源码包根目录下配置数据集路径。

   ```
   mkdir datasets
   ```

2. 获取数据集。

   用户自行获取原始数据集DOTAv1，将数据集上传到服务器任意路径下并解压到datasets文件夹下，数据集目录结构参考如下所示：

   ```
    ├── DOTAv1
    │   ├── images
    │          ├── test
    │          ├── train
    │          ├── val
    │   ├── labels
    │          ├── train
    │          ├── train_original
    │          ├── val
    │          ├── val_original
    │          ├── train.cache
    │          ├── val.cache
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

3. 将ultralytics/cfg/datasets/DOTAv1.yaml文件中的path改成DOTAv1数据集的绝对路径

     ```
      path: /your/path/to/datasets/DOTAv1
     ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   - 单卡训练性能

     启动单卡训练。

     ```
     bash test/train_yolov8_performance_1p.sh  # yolov8 1p_performance
     ```

   - 单卡训练精度

     启动单卡训练。

     ```
     bash test/train_yolov8_full_1p.sh  # yolov8 1p_full
     ```
   - 八卡训练性能

     启动八卡训练。

     ```
     bash test/train_yolov8_performance_8p.sh  # yolov8 8p_performance
     ```

   - 八卡训练精度

     启动八卡训练。

     ```
     bash test/train_yolov8_full_8p.sh  # yolov8 8p_full
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径    
   --batch-size                        //训练批次大小
   --epochs                            //重复训练次数
   --weights                           //初始权重路径
   --data_shuffle                      //打乱数据
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  yolov8训练结果展示表

| NAME     | MODE | mAP50 |  mAP50-95   | FPS | Torch_Version |
| :-----:  | :---:  | :---:  |:------:|:--------:| :------: |
| 1p-竞品A  | fp32 | 0.492 | 0.351 |    198.33    |    2.1 |
| 1p-NPU | fp32 | 0.491 |  0.351   |    171.66    |      2.1      |
| 8p-竞品A  | fp32 | 0.501 | 0.359 |    1587.60    |    2.1 |
| 8p-NPU | fp32 | 0.504 |  0.361   |    943.00    |      2.1      |

# 版本说明

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
