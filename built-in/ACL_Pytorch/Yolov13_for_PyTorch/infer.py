# Copyright (c) 2026 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse

import torch
import torch_npu

from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="yolov13 infer")
    parser.add_argument("--pth", type=str, help="path to checkpoint file")
    parser.add_argument("--dataset", type=str, help="path to description file of dataset")
    parser.add_argument("--batchsize", type=int, help="batch size for inference")
    args = parser.parse_args()
    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = YOLO(args.pth)

    with torch.no_grad():
        model.val(data=args.dataset, batch=args.batchsize, half=True, imgsz=640)