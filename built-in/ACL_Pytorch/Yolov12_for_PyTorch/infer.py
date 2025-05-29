# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import math

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

from ultralytics.nn.modules.block import AAttn
from ultralytics import YOLO


def aattn_forward(AAttn_obj, x):
    """Processes the input tensor 'x' through the area-attention(use FA from torch_npu instead of vanilla attention)"""
    B, C, H, W = x.shape
    N = H * W

    qk = AAttn_obj.qk(x).flatten(2).transpose(1, 2)
    v = AAttn_obj.v(x)
    pp = AAttn_obj.pe(v)
    v = v.flatten(2).transpose(1, 2)

    if AAttn_obj.area > 1:
        qk = qk.reshape(B * AAttn_obj.area, N // AAttn_obj.area, C * 2)
        v = v.reshape(B * AAttn_obj.area, N // AAttn_obj.area, C)
        B, N, _ = qk.shape
    q, k = qk.split([C, C], dim=2)

    q = q.view(B, N, AAttn_obj.num_heads, AAttn_obj.head_dim)
    k = k.view(B, N, AAttn_obj.num_heads, AAttn_obj.head_dim)
    v = v.view(B, N, AAttn_obj.num_heads, AAttn_obj.head_dim)

    x = torch_npu.npu_prompt_flash_attention(
        q.contiguous().half(),
        k.contiguous().half(),
        v.contiguous().half(),
        input_layout='BSND',
        num_heads=AAttn_obj.num_heads,
        scale_value=1 / math.sqrt(AAttn_obj.head_dim)
    )
    if AAttn_obj.area > 1:
        x = x.reshape(B // AAttn_obj.area, N * AAttn_obj.area, C)
        B, N, _ = x.shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return AAttn_obj.proj(x + pp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="yolov12 infer")
    parser.add_argument("--pth", type=str, help="path to checkpoint file")
    parser.add_argument("--dataset", type=str, help="path to description file of dataset")
    parser.add_argument("--batchsize", type=int, help="batchsize")
    args = parser.parse_args()

    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = YOLO(args.pth)

    for module in model.modules():
        # use FA in forward instead of vanilla attention
        if isinstance(module, AAttn):
            module.forward = aattn_forward.__get__(module)

    # 添加torchair适配代码
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)
    model = torch.compile(model, dynamic=True, fullgraph=True, backend=npu_backbend)

    with torch.no_grad():
        model.val(data=args.dataset, batch=args.batchsize, half=True, imgsz=640, device='npu')