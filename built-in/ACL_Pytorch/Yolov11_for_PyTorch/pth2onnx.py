# Copyright (c) 2025 Huawei Technologies Co., Ltd
# this software is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', default="yolo11n.pt", help='pt file')
    parser.add_argument('--batch', type=int, default=1, help='batch size for ONNX export')

    args = parser.parse_args()

    model = YOLO(args.pt)
    onnx_model = model.export(format="onnx", dynamic=False, simplify=False, opset=12, batch=args.batch)


    # Rename the ONNX file to include batchsize information

    # Get the original ONNX file path
    original_path = Path(onnx_model)
    base_name = original_path.stem 

    # Set the new file name to include batchsize information
    if f"_bs{args.batch}" not in base_name:
        new_name = f"{base_name}_bs{args.batch}.onnx"
        new_path = original_path.parent / new_name

        if original_path.exists():
            os.rename(original_path, new_path)
            print(f"ONNX file has been renamed: {original_path} -> {new_path}")
            onnx_model = str(new_path)

    print(f"export to {onnx_model} (batchsize={args.batch})")


if __name__ == '__main__':
    main()