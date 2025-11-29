# 数据集下载与准备

## 整体介绍
本文档记录了modelzoo-GPL/built-in/ACL_Pytorch/目录下各Yolo版本所需数据集的下载与准备方式。

**数据集与使用模型对照表**
|    数据集  |     目录名称   |  使用模型(任务)     |
|---------------|---------------|---------------|
|      coco2017 val  |     coco     | yolov3、yolov5、yolov7、yolov8、yolov11(detect/segment)、yolov12
|    ImageNet val  |   imagenet   | yolov11(classify)|
|   coco2017-pose val  |  coco-pose  | yolov11(pose) |
|   DOTAv1  |  DOTAv1  | yolov11(obb) |



**数据集下载链接一览表**

|    数据集   | 文件  |    下载地址          |
|----------------|-------|-----------------------------|
|   coco2017 val | coco2017labels-segments.zip |[点击下载](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip)|
|    | val2017.zip |[点击下载](http://images.cocodataset.org/zips/val2017.zip)|
|   ImageNet val| ILSVRC2012_img_val.tar|[点击下载](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)|
|   coco2017-pose val| ILSVRC2012_img_val.tar|[点击下载](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip)|
|   | val2017.zip|同coco2017 val数据集val2017.zip文件|
| DOTAv1  | DOTAv1.zip|[点击下载](https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip)|

## 下载与准备详解
### coco2017 val 数据集
```bash
#进入项目目录
cd ${YOLO_PATH}

# 创建数据集目录
mkdir -p ../datasets/coco/images

cd ../datasets

# 下载并解压标注文件
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip
unzip coco2017labels-segments.zip

cd coco

#下载并解压图像文件
wget http://images.cocodataset.org/zips/val2017.zip    # 1G, 5k images
unzip val2017.zip -d images/
mkdir -p images/train2017  # 创建空的训练集目录



#创建图像列表文件 
cd ../datasets/coco

# 创建验证图像列表 (对应coco.yaml中的 val: val2017.txt)
ls images/val2017/*.jpg | sed 's/^/images\/val2017\//g' > val2017.txt
```
最终数据集结构如下
```
datasets/
├── labels/
│   ├── train2017/
│   └── val2017/
└── coco/
    ├── images/
    │   ├── train2017/
    │   └── val2017/ 
    └── val2017.txt
```
### coco2017-pose val 数据集
```bash
#创建文件夹
cd ${YOLO_PATH}
mkdir -p datasets/coco-pose/images
cd datasets/

#下载标签并解压
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip
unzip coco2017labels-pose.zip

#下载图片并解压
cd coco-pose/images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

#创建一个空的训练集
mkdir train2017

#创建图像文件列表
cd ..
ls labels/val2017/*.txt | sed 's/labels\/val2017\///g' | sed 's/\.txt/.jpg/g' | sed 's/^/images\/val2017\//g' > val2017.txt

```
最终数据集结构如下
```
.
├── images
│   ├── train2017
│   ├── val2017
│   └── val2017.zip
├── labels
│   ├── train2017
│   └── val2017
└── val2017.txt
```
### ImageNet val 数据集

```bash
cd ${YOLO_PATH}

mkdir -p ../datasets/imagenet/val
mkdir -p ../datasets/imagenet/train
mkdir -p ../datasets/imagenet/test
cd ../datasets/imagenet/val

# 下载并解压验证集
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_img_val.tar

# 执行预处理脚本
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
最终数据集结构如下
```
imagenet/
  ├── train/
  ├── test/
  └── val/
      ├── n01440764/
      │   ├── ILSVRC2012_val_00000293.JPEG
      │   └── ...
      ├── n01443537/
      │   ├── ILSVRC2012_val_00000338.JPEG
      │   └── ...
      └── ...
```
### DOTAv1 数据集
```bash
cd ${YOLO_PATH}

# 创建DOTAv1数据集根目录
mkdir -p ../datasets/DOTAv1/images
cd ../datasets/DOTAv1

# 下载DOTAv1数据集并解压 (约2GB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip
unzip DOTAv1.zip
```

最终目录结构
```
DOTAv1/
├── images/
│   ├── train/
│   │   ├── P0000.png
│   │   ├── P0001.png
│   │   └── ...
│   ├── val/
│   │   ├── P1411.png
│   │   ├── P1412.png
│   │   └── ...
│   └── test/
│       ├── P1869.png
│       ├── P1870.png
│       └── ...
└── labels/
   ├── train/
   │   ├── P0000.txt
   │   ├── P0001.txt
   │   └── ...
   ├── val/
   │   ├── P1411.txt
   │   ├── P1412.txt
   │   └── ...
   └── test/
```
