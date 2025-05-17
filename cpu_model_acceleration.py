# Copyright (c) 2025 Intel Corporation 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications v1:
# - Added command-line argument parsing for model configuration and quantization settings.
# - Integrated OpenVINO model conversion and quantization processes.
# - Implemented accuracy control and benchmarking for different model precisions (FP32, FP16, INT8).
# - Enhanced the script with debug mode and additional logging for easier troubleshooting.
# - Updated the dataset handling to include both validation and training subsets for quantization calibration.


from functools import partial
from pathlib import Path
import argparse

import openvino as ov
import torch
from torchvision import datasets

import timm
import nncf
import os
from pdb import set_trace as st
from nncf import compress_weights, CompressWeightsMode
from copy import deepcopy
from nncf.parameters import ModelType
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from utilities import val_transform, transform_fn, validate_pt, validate_ov, run_benchmark, get_model_size

def parse_arguments():
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description='Quantize a PyTorch model with OpenVINO.')
    
    # 添加命令行参数
    parser.add_argument('--model_name', type=str, required=True, default='tnt_s_patch16_224',
                        choices=['tnt_s_patch16_224', 'deit_base_patch16_224',
                                 'swin_tiny_patch4_window7_224', 'resnet50', 'mobilenetv2_100'],
                        help='Specify the model name. Choose from: tnt_s_patch16_224, deit_base_patch16_224, '
                             'swin_tiny_patch4_window7_224, resnet50, mobilenetv2_100')
    parser.add_argument('--accuracy_control', type=bool, default=False,
                        help='Enable or disable accuracy control during quantization.')
    parser.add_argument('--transformer_model', type=bool, default=False,
                        help='Specify whether the model is a Transformer model.')
    parser.add_argument('--weight_only', type=bool, default=False,
                        help='Enable weight-only compression.')
    parser.add_argument('--dataset_path', type=str, default='/data/imagenet',
                        help='Specify the path to the dataset. Default is /data/imagenet.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Specify the batch size for data loading. Default is 128.')
    parser.add_argument('--baseline_precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Specify the baseline precision for model inference. Choose from: fp32, fp16. Default is fp32 since in this setting both fp32 and fp16 results are exported.')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Enable debug mode for additional logging and checks.')
    
    # 解析命令行参数
    args = parser.parse_args()
    return args

def main():
    # 解析命令行参数
    args = parse_arguments()
    # 使用解析的参数
    ACC_CONTROL = args.accuracy_control
    IS_TRANSFORMER = ModelType.TRANSFORMER if args.transformer_model else None
    WEIGHT_ONLY = args.weight_only
    model_name = args.model_name
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    precision = args.baseline_precision
    DEBUG = args.debug
    
    ROOT = Path(__file__).parent.resolve()
    print(f"ROOT: {ROOT}")
    
    torch_model = timm.create_model(
        model_name,  # 确保名称完全匹配
        pretrained=True,       # 启用自动加载
        num_classes=1000       # 保持与预训练一致
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if precision == "fp16":
        torch_model.half().to(device)
        dummy_input = torch.randn(1, 3, 224, 224).half()
    else:
        torch_model.to(device)
        dummy_input = torch.randn(1, 3, 224, 224)
    torch_model.eval()

    if not ACC_CONTROL:
        val_dataset = datasets.ImageFolder(
            root=os.path.join(dataset_path , "val"),
            transform=val_transform,
        )
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        subset_size = 300 // batch_size
        calibration_dataset = nncf.Dataset(val_data_loader, partial(transform_fn, device=device))
        if WEIGHT_ONLY:
            torch_quantized_model = compress_weights(deepcopy(torch_model), 
                                             mode=CompressWeightsMode.INT8_ASYM,  
                                             dataset=calibration_dataset, gptq=None,
                                             group_size=-1,
                                             )
        else:
            torch_quantized_model = nncf.quantize(torch_model, calibration_dataset, subset_size=subset_size,
                                        model_type=IS_TRANSFORMER)
        ov_model = ov.convert_model(torch_model.cpu(), example_input=dummy_input)
        ov_quantized_model = ov.convert_model(torch_quantized_model.cpu(), example_input=dummy_input)
    else:
        train_dataset = ImageFolder(
        root=os.path.join(dataset_path , "train"),
        transform=val_transform
        )
        num_samples =300
        indices = random.sample(range(len(train_dataset)), num_samples)
        # 创建子集
        calibration_subset = Subset(train_dataset, indices)
        calibration_loader = DataLoader(
            calibration_subset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        device = "cpu"
        calibration_dataset = nncf.Dataset(calibration_loader, partial(transform_fn, device=device))
        validation_dataset = nncf.Dataset(val_data_loader, partial(transform_fn, device=device))
        ov_model = ov.convert_model(torch_model.cpu(), example_input=dummy_input)
        print("Quantize_with_accuracy_control!")
        ov_quantized_model = nncf.quantize_with_accuracy_control( #Note this function can only use openvino model!
        ov_model,
        calibration_dataset=calibration_dataset,
        validation_dataset=validation_dataset,
        validation_fn=validate_ov,
        max_drop=0.015,
        drop_type=nncf.DropType.ABSOLUTE,
        model_type=IS_TRANSFORMER
    )
    if DEBUG:
        print(id(torch_quantized_model)==id(torch_model))
        print(torch_quantized_model)
        st()
    ###############################################################################
    # Benchmark performance, calculate compression rate and validate accuracy
    print(f"[0/10] Model name: {model_name}")
    print("[1/10] Validate OpenVINO INT8 model:")
    int8_top1 = validate_pt(ov_quantized_model, val_data_loader)
    print(f"Accuracy @ top1: {int8_top1:.3f}")

    if precision == "fp32": #This mode export the comprehensive info. This mode "ov_model" is fp32
        print("[2/10] Validate OpenVINO FP32 model:")
        fp32_top1 = validate_pt(ov_model, val_data_loader)
        print(f"Accuracy @ top1: {fp32_top1:.3f}")
    
        fp32_ir_path = os.path.join(ROOT,model_name,"_fp32.xml")
        ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
        print(f"[3/10] Save FP32 model: {fp32_ir_path}")
        fp32_model_size = get_model_size(fp32_ir_path)

        fp16_ir_path = os.path.join(ROOT,model_name,"_fp16.xml")
        ov.save_model(ov_model, fp16_ir_path, compress_to_fp16=True)
        print(f"[4/10] Save FP16 model: {fp16_ir_path}")
        fp16_model_size = get_model_size(fp16_ir_path)

        ov_model_fp16 = ov.Core().read_model(fp16_ir_path) # openvino.Model object
        print("[5/10] Validate OpenVINO FP16 model:")
        fp16_top1 = validate_pt(ov_model_fp16, val_data_loader)
        print(f"Accuracy @ top1: {fp16_top1:.3f}")

        int8_ir_path = os.path.join(ROOT,model_name,"_int8.xml")
        ov.save_model(ov_quantized_model, int8_ir_path)
        print(f"[6/10] Save INT8 model: {int8_ir_path}")
        int8_model_size = get_model_size(int8_ir_path)
    
        print("[7/10] Benchmark FP32 model:")
        fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 224, 224])
        print("[8/10] Benchmark FP16 model:")
        fp16_fps = run_benchmark(fp16_ir_path, shape=[1, 3, 224, 224])
        print("[9/10] Benchmark INT8 model:")
        int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 224, 224])

        print("[10/10] Report:")
        print(f"Accuracy drop compared to the fp32 baseline: {fp32_top1 - int8_top1:.3f}")
        print(f"Relative accuracy drop compared to the fp32 baseline: {(fp32_top1 - int8_top1)/1.0/fp32_top1:.3f}")
        print(f"Model compression rate compared to the fp32 baseline:: {fp32_model_size / int8_model_size:.3f}")
        print(f"Performance speed up (throughput mode) compared to the fp32 baseline:: {int8_fps / fp32_fps:.3f}")
        print('*'*10)
        print(f"Accuracy drop compared to the fp16 baseline: {fp16_top1 - int8_top1:.3f}")
        print(f"Relative accuracy drop compared to the fp16 baseline: {(fp16_top1 - int8_top1)/1.0/fp16_top1:.3f}")
        print(f"Model compression rate compared to the fp16 baseline:: {fp16_model_size / int8_model_size:.3f}")
        print(f"Performance speed up (throughput mode) compared to the fp16 baseline:: {int8_fps / fp16_fps:.3f}")
        del fp32_top1, fp32_ir_path, fp32_model_size, fp16_ir_path, fp16_model_size
        del ov_model_fp16, fp16_top1, int8_ir_path, int8_model_size, fp32_fps, fp16_fps, int8_fps
    else: 
        #This mode "ov_model" is fp16
        print("[2/7] Validate OpenVINO FP16 model:")
        fp16_top1 = validate_pt(ov_model, val_data_loader)
        print(f"Accuracy @ top1: {fp16_top1:.3f}")

        fp16_ir_path = os.path.join(ROOT,model_name,"_fp16.xml")
        ov.save_model(ov_model, fp16_ir_path, compress_to_fp16=True)
        print(f"[3/7] Save FP16 model: {fp16_ir_path}")
        fp16_model_size = get_model_size(fp16_ir_path)

        int8_ir_path = os.path.join(ROOT,model_name,"_int8.xml")
        ov.save_model(ov_quantized_model, int8_ir_path)
        print(f"[4/7] Save INT8 model: {int8_ir_path}")
        int8_model_size = get_model_size(int8_ir_path)

        print("[5/7] Benchmark FP16 model:")
        fp16_fps = run_benchmark(fp16_ir_path, shape=[1, 3, 224, 224])
        print("[6/7] Benchmark INT8 model:")
        int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 224, 224])

        print("[7/7] Report:")
        print(f"Accuracy drop compared to the fp16 baseline: {fp16_top1 - int8_top1:.3f}")
        print(f"Relative accuracy drop compared to the fp16 baseline: {(fp16_top1 - int8_top1)/1.0/fp16_top1:.3f}")
        print(f"Model compression rate compared to the fp16 baseline:: {fp16_model_size / int8_model_size:.3f}")
        print(f"Performance speed up (throughput mode) compared to the fp16 baseline:: {int8_fps / fp16_fps:.3f}")


if __name__ == "__main__":
    main()