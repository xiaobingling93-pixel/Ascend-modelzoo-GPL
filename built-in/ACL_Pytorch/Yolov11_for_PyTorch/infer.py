# Copyright (c) 2025 Huawei Technologies Co., Ltd
# this software is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import re
from pathlib import Path

import torch
import torch_npu

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

TASK2MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}

TASK2DATA = {
    "detect": "coco.yaml",
    "segment": "coco.yaml",
    "classify": "imagenet",
    "pose": "coco-pose.yaml",
    "obb": "DOTAv1.yaml",
}

# Change DATASETs_DIR to current project
current_project_root = Path(__file__).resolve().parent
SETTINGS["datasets_dir"] = str(current_project_root / "datasets")
print(f"Modified DATASETS_DIR to: {SETTINGS['datasets_dir']}")



# Determine whether om infer is supported
try:
    from ais_bench.infer.interface import InferSession
    OM_AVAILABLE = True
except ImportError:
    OM_AVAILABLE = False
    print("Warning: ais_bench not available, OM model inference disabled")


def get_auto_model(task):
    """Return model type from task type"""
    if task not in TASK2MODEL:
        raise ValueError(f"Invalid task '{task}'. Valid tasks are: {list(TASK2MODEL.keys())}")
    return TASK2MODEL[task]


def is_om_model(model_path):
    return Path(model_path).suffix.lower() == '.om'


def get_auto_dataset(task):
    """Return dataset config from task type"""
    return TASK2DATA.get(task, None)


def get_default_imgsz(task):
    """Return default image size from task type"""
    if task == "classify":
        return 224
    elif task == "obb":
        return 1024
    else:
        return 640


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="yolov11 validation with NPU support and auto model selection")
    parser.add_argument("--pth", type=str, default=None, help="model weights path (optional, will auto-select by task)")
    parser.add_argument("--dataset", type=str, default=None, help="dataset yaml path (optional, will auto-select by task)")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment", "pose", "classify", "obb"], help="task type")
    parser.add_argument("--device", type=str, default="0", help="device (0 for auto, npu:0, cpu, cuda:0)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="dataset split to use (val, test, train)")
    args = parser.parse_args()

    torch_npu.npu.set_compile_mode(jit_compile=False)

    # Auto model selection if not provided
    if args.pth is None:
        auto_model = get_auto_model(args.task)
        print(f"No model specified, auto-selecting for {args.task} task: {auto_model}")
        args.pth = auto_model
    else:
        print(f"Using specified model: {args.pth}")

    # Check model type (PyTorch vs OM)
    use_om = is_om_model(args.pth)
    if use_om:
        if not OM_AVAILABLE:
            raise ImportError("OM model specified but ais_bench not available. Please install ais_bench.")
        print(f"Detected OM model: {args.pth}")
    else:
        print(f"Detected PyTorch model: {args.pth}")

    # Auto dataset selection if not provided
    if args.dataset is None:
        auto_dataset = get_auto_dataset(args.task)
        if auto_dataset:
            print(f"No dataset specified, auto-selecting for {args.task} task: {auto_dataset}")
            args.dataset = auto_dataset
        else:
            print(f"No auto-dataset available for {args.task} task")
            args.dataset = None

    if args.dataset is None:
        raise ValueError("Dataset must be specified either via --dataset or auto-selected for the task")

    # Load model
    if use_om:
        print(f"OM model detected: {args.pth}")
        print("Using OM model for inference (AutoBackend ACL support)")

        # Check if corresponding PyTorch model exists for metadata fallback
        
        pt_model_path = re.sub(r'_bs\d+\.om$', '.pt', args.pth)
        if Path(pt_model_path).exists():
            print(f"PyTorch reference model available: {pt_model_path}")
        else:
            raise FileNotFoundError(f"PyTorch reference model not found: {pt_model_path}")

        # Load OM model directly via AutoBackend
        print(f"Loading OM model: {args.pth}")
    else:
        print(f"Loading PyTorch model: {args.pth}")

    model = YOLO(args.pth)

    # Verify actual model type used
    if hasattr(model.model, 'om') and model.model.om:
        print(f"Confirmed: Using OM model backend")
    elif hasattr(model.model, 'pt') and model.model.pt:
        print(f"Confirmed: Using PyTorch model backend")
    else:
        print(f"Confirmed: Using {type(model.model).__name__} backend")

    # Verify model matches task
    if args.task and hasattr(model, 'task') and model.task != args.task:
        print(f"Warning: Model task is '{model.task}' but requested task is '{args.task}'")
        print(f"Auto-adjusting to model task: {model.task}")
        args.task = model.task

    # Prepare validation arguments exactly like yolo CLI
    val_args = {
        "data": args.dataset,
        "batch": args.batchsize,
        "half": True, 
        "imgsz": get_default_imgsz(args.task),
        "device": args.device,
        "split": args.split,
        "plots": True,
        "save_json": False,
    }

    # Add task-specific arguments
    if args.task == "segment":
        val_args["overlap_mask"] = True
    elif args.task == "pose":
        val_args["pose"] = True

    print(f"\nStarting {args.task} validation:")
    print(f"  Model: {args.pth}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Image size: {val_args['imgsz']} ({'classification' if args.task == 'classify' else 'detection'} size)")
    print(f"  Device: {args.device}")
    print(f"  Half precision: True")

    # Run validation with NPU acceleration
    with torch.no_grad():
        results = model.val(**val_args)

    print("\nValidation completed!")
    print(f"Results: {results}")

    # Print key metrics
    if hasattr(results, 'box') and results.box:
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"mAP50: {results.box.map50:.4f}")
    if hasattr(results, 'segment') and results.segment:
        print(f"Segmentation mAP50-95: {results.segment.map:.4f}")
    if hasattr(results, 'pose') and results.pose:
        print(f"Pose mAP50-95: {results.pose.map:.4f}")
