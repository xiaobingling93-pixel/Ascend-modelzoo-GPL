# Yolov5_for_PyTorch_v6.0

- [Yolov5\_for\_PyTorch\_v6.0](#yolov5_for_pytorch_v60)
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
  - [变更](#变更)
  - [FAQ](#faq)
- [YOLOv5-离线推理指导](#yolov5-离线推理指导)
- [概述](#概述-1)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [安装依赖](#安装依赖)
  - [准备数据集](#准备数据集-1)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
        - [\* 如果有多卡推理的需求，请跳过该步骤，om\_val.py该脚本不支持多卡推理](#-如果有多卡推理的需求请跳过该步骤om_valpy该脚本不支持多卡推理)
- [模型推理性能\&精度](#模型推理性能精度)
- [FAQ](#faq-1)

# 概述

## 简述
YOLO是一个经典的物体检测网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。YOLOv5于2020.05.27首次发布，截至2020.12.01仍在更新，目前NPU适配的版本为Yolov5 Tag=v6.0。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5.git
  Tag=v6.0
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelzoo-GPL
  code_path=built-in/PyTorch/Official/cv/object_detection
  ```

# 准备训练环境

## 准备环境

- 推荐使用最新的版本准备训练环境。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1.1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v7.0.0-pytorch2.1.0 </td>
    </tr>
  </table>

- 三方库依赖如下表所示。

  **表 2**  三方库依赖表

  | Torch_Version | 三方库依赖版本                                 |
  |:-------------:| :----------------------------------------------------------: |
  |  PyTorch 2.1  | pillow==9.1.0 |
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  
  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集coco2017，将数据集上传到服务器任意路径下并解压，数据集目录结构参考如下所示：

   ```
   ├── coco #根目录
        ├── train2017 #训练集图片，约118287张
        ├── val2017 #验证集图片，约5000张
        └── annotations #标注目录
        		  ├── instances_train2017.json #对应目标检测、分割任务的训练集标注文件
        		  ├── instances_val2017.json #对应目标检测、分割任务的验证集标注文件
        		  ├── captions_train2017.json
        		  ├── captions_val2017.json
        		  ├── person_keypoints_train2017.json
        		  └── person_keypoints_val2017.json
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。
2. 生成yolov5专用标注文件。

   （1）将代码仓中coco/coco2yolo.py和coco/coco_class.txt拷贝到coco数据集**根目录**。

   （2）运行coco2yolo.py。

   ```
   python3 coco2yolo.py
   ```

   （3）运行上述脚本后，将在coco数据集**根目录**生成train2017.txt和val2017.txt。
3. 在源码包根目录下配置数据集路径。

   ```
   mkdir datasets
   ln -s coco_path ./datasets/coco  # coco_path为数据集实际路径
   ```

  - 编译安装torchvision
  
    ***为了更快的推理性能，请编译安装而非直接安装torchvision***


     ```
      git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
      cd vision
      python setup.py bdist_wheel
      pip3 install dist/*.whl
     ```
***编译安装过程中会卸载已安装的torch，编译完成后，请重新安装torch_npu***

4. 编译安装Opencv-python。

   为了获得最好的图像处理性能，**请编译安装opencv-python而非直接安装**。编译安装步骤如下：

   ```
   export GIT_SSL_NO_VERIFY=true
   git clone https://github.com/opencv/opencv.git
   cd opencv
   mkdir -p build
   cd build
   cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/root/miniconda3/envs/yolov5/bin/python3.8 -D PYTHON3_INCLUDE_DIR=/root/miniconda3/envs/yolov5/include/python3.8 -D PYTHON3_LIBRARY=/root/miniconda3/envs/yolov5/lib/libpython3.8.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/root/miniconda3/envs/yolov5/lib/python3.8/site-packages/numpy/core/include -D   PYTHON3_PACKAGES_PATH=/root/miniconda3/envs/yolov5/lib/python3.8/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/root/miniconda3/envs/yolov5/bin/python3.8 ..
   make -j$nproc
   make install
   ```
   
注：

上述cmake命令中的路径，请自行替换为实际conda虚拟环境或者系统环境的Python路径

编译安装过程中会刷新numpy的版本，该版本的numpy无法运行本项目！

请在编译安装OpenCV后，将numpy刷回1.23.0版本

```shell
pip install numpy==1.23.0
```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash test/train_yolov5s_performance_1p.sh  # yolov5s 1p_performance
     bash test/train_yolov5m_performance_1p.sh  # yolov5m 1p_performance
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash test/train_yolov5s_performance_8p.sh  # yolov5s 8p_performance
     bash test/train_yolov5m_performance_8p.sh  # yolov5m 8p_performance
     bash test/train_yolov5s_full_8p.sh  # yolov5s 8p_accuracy
     bash test/train_yolov5m_full_8p.sh  # yolov5m 8p_accuracy
     bash test/train_yolov5m_full_8p_high_preci.sh  # yolov5m 8p_high_precision 
     ```

   - 纯FP32计算

     启动单卡训练
     ```
     bash test/train_yolov5s_fp32_performance_1p.sh  # yolov5s 1p_fp32_performance
     ```
     启动多卡训练
     ```
     bash test/train_yolov5s_fp32_performance_8p.sh  # yolov5s 8p_fp32_performance
     bash test/train_yolov5s_fp32_full_8p.sh  # yolov5s 8p_fp32_accuracy
     ```

   - 纯HF32计算

     启动单卡训练
     ```
     bash test/train_yolov5s_fp32_performance_1p.sh --hf32   # yolov5s 1p_hf32_performance
     ```
     启动多卡训练
     ```
     bash test/train_yolov5s_fp32_performance_8p.sh --hf32  # yolov5s 8p_hf32_performance
     bash test/train_yolov5s_fp32_full_8p.sh --hf32     # yolov5s 8p_hf32_accuracy

   - NPU 多机多卡训练指令
   
     启动多机多卡训练。
     ```
     bash test/train_yolov5s_performance_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     bash test/train_yolov5m_performance_cluster.sh --data_path=数据集路径 --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     ```
     ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练。
  
     
   - 在线推理
     启动在线推理。
     ```
     bash ./test/train_yolov5s_eval.sh #在线推理
     ```
     

   --data_path参数填写数据集路径，需写到数据集的一级目录。


   模型训练脚本参数说明如下。

   ```
   公共参数：
   --conda_name                        //conda名，不传默认没有，传了进入conda执行
   --data                              //数据集路径
   --workers                           //加载数据进程数     
   --batch-size                        //训练批次大小
   --epochs                            //重复训练次数
   --weights                           //初始权重路径
   --rect                              //矩形训练
   --nosave                            //保存最后一个权重
   --noval                             //验证最后一个epoch
   --artifact_alias                    //数据集版本
   --save-period                       //权重保存
   --native_amp                        //使用torch amp进行混合精度训练，如不配置默认使用apex
   --half                              //eval执行脚本中参数，如配置默认使用混合精度计算
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  yolov5m训练结果展示表

| NAME     | mAP0.5 |  FPS   | AMP_Type | Torch_Version |
| :-----:  | :---:  |:------:|:--------:| :------: |
| 1p-竞品A  | - | 181 |    O1    |    1.8 |
| 8p-竞品A | 64.1 |  1264  |    O1    |      1.8      |
| 1p-NPU | - | 283.3  |    O1    |     1.11      |
| 8p-NPU | 63.6 | 2208.5 |    O1    |     1.11      |
| 1p-NPU | - |  281   |    O1    |      2.1      |
| 8p-NPU | 63.6 | 2106.6 |    O1    |      2.1      |

**表 4**  yolov5m高精度8p训练结果展示表

| NAME     | mAP0.5 |  FPS | AMP_Type | Torch_Version |
| :-----:  | :---:  | :--: | :------: | :------: |
| 8p-竞品A | 64.1 |  1264  |    O1    |      1.8      |
| 8p-NPU |  64.5 | 2208.5 |  O1     |     1.11      |
| 8p-NPU |  64.5 | 2106.6 |   O1     |      2.1      |

**表 5**  yolov5s训练结果展示表

| NAME     | mAP0.5~0.95 |  FPS   | AMP_Type | Torch_Version | Architecture |  Device_Type  |
| :-----:  | :---:  |:------:| :-----:  |:-------------:| :-----: |:-------------:|
| 1p-NPU | 35.5 | 419.7 |   FP16   |     1.11      | Arm | Atlas 800T A2 |
| 8p-NPU | 35.5 | 3082.4  |   FP16   |     1.11      | Arm | Atlas 800T A2 |
| 1p-NPU | 35.5 | 428.5 |   FP16   |      2.1      | Arm | Atlas 800T A2 |
| 8p-NPU | 35.5 | 3006.9  |   FP16   |      2.1      | Arm | Atlas 800T A2 |
| 1p-NPU | 35.5 | 221.6 |   FP32   |     1.11      | Arm | Atlas 800T A2 |
| 8p-NPU | 35.5 | 1724.8  |   FP32   |     1.11      | Arm | Atlas 800T A2 |
| 1p-NPU | 35.5 | 224 |   FP32   |      2.1      | Arm | Atlas 800T A2 |
| 8p-NPU | 35.5 | 1742.3    |   FP32   |      2.1      | Arm | Atlas 800T A2 |
| 1p-NPU | 35.5 | 221.3  |   HF32   |     1.11      | Arm | Atlas 800T A2 |
| 8p-NPU | 35.5 | 1741.7 |   HF32   |     1.11      | Arm | Atlas 800T A2 |
| 1p-NPU | 35.5 | 222.5  |   HF32   |      2.1      | Arm | Atlas 800T A2 |
| 8p-NPU | 35.5 | 1716.6 |   HF32   |      2.1      | Arm | Atlas 800T A2 |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| yolov5m 6.0-8p-竞品 | FP16 | 1264 |
| yolov5m 6.0-8p-Atlas 900 A2 PoDc | FP16 | 2295.1 |
| yolov5s 6.0-8p-竞品 | FP16 | 3006.9 |
| yolov5s 6.0-8p-Atlas 900 A2 PoDc | FP16 | 3097.8 |

# 版本说明

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

## 变更

2023.02.16：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

1. 训练过程中若遇到该问题`wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key])`，在不能获取到key的情况下，请卸载三方库`wandb`，再进行训练。
