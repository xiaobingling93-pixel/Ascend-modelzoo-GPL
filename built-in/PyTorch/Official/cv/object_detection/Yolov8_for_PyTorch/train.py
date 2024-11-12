from ultralytics import YOLO

import os
import sys
import torch
import torch_npu
import argparse
from pathlib import Path
from torch_npu.contrib import transfer_to_npu
torch.npu.config.allow_internal_format = True

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov8n-obb.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'ultralytics/cfg/models/v8/yolov8-obb.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/DOTAv1.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_shuffle', default=True, action="store_false")
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='npu device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    # Load a model
    model = YOLO(opt.cfg).load(opt.weights)

    # Train the model
    model.train(
        data=opt.data,  # path to dataset YAML
        epochs=opt.epochs,  # number of training epochs
        imgsz=opt.imgsz,  # training image size
        data_shuffle=opt.data_shuffle,
        device=opt.device,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )