#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path> [batchsize] [chip_name]"
    echo "Example: $0 yolo11n_bs2.onnx"
    echo "Example: $0 yolo11n-seg_bs4.onnx 4 310P3"
    exit 1
fi

model_path=$1
chip_name=${3:-310P3}


extract_batchsize_from_filename() {
    local filename="$1"
    if [[ $filename =~ _bs([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

if [ ! -f "$model_path" ]; then
    echo "Error: Model file not found: $model_path"
    exit 1
fi

# Parse batchsize from filename
model_name=$(basename "$model_path" .onnx)
echo "Processing model: $model_name"
if [ $# -ge 2 ]; then
    batchsize=$2
    echo "Using user-specified batch size: $batchsize"
else
    parsed_batch=$(extract_batchsize_from_filename "$model_name")
    if [ -n "$parsed_batch" ]; then
        batchsize=$parsed_batch
        echo "Auto-detected batch size from filename: $batchsize"
    else
        echo "Error: No batch size specified and could not detect batch size from filename"
        echo "Usage: $0 <model_path> [batchsize] [chip_name]"
        echo "Please either specify batchsize as parameter or ensure filename contains '_bs<number>' pattern"
        exit 1
    fi
fi


# Identify the type of task from filename
if [[ $model_name == *"-seg"* ]]; then
    task="segment"
    imgsz=640
    echo "Task detected: Segmentation, Image size: ${imgsz}x${imgsz}"
elif [[ $model_name == *"-cls"* ]]; then
    task="classify"
    imgsz=224
    echo "Task detected: Classification, Image size: ${imgsz}x${imgsz}"
elif [[ $model_name == *"-pose"* ]]; then
    task="pose"
    imgsz=640
    echo "Task detected: Pose, Image size: ${imgsz}x${imgsz}"
elif [[ $model_name == *"-obb"* ]]; then
    task="obb"
    imgsz=1024
    echo "Task detected: Oriented Bounding Box, Image size: ${imgsz}x${imgsz}"
else
    task="detect"
    imgsz=640
    echo "Task detected: Detection (default), Image size: ${imgsz}x${imgsz}"
fi

if ! command -v atc &> /dev/null; then
    echo "Error: ATC tool not found. Please ensure CANN toolchain is properly installed and in PATH."
    exit 1
fi

# Generate output filename
if [[ $model_name =~ _bs[0-9]+$ ]]; then
    output_name="${model_name}"
else
    output_name="${model_name}_bs${batchsize}"
fi
echo "Output will be saved as: ${output_name}.om"

echo "Conversion parameters:"
echo "  Framework: 5 (ONNX)"
echo "  Model: $model_path"
echo "  Input format: NCHW"
echo "  Input shape: images:${batchsize},3,${imgsz},${imgsz}"
echo "  Output type: FP16"
echo "  Output: ${output_name}.om"
echo "  Chip version: Ascend${chip_name}"
echo "  Batch size: $batchsize"
echo ""

echo "Starting conversion..."
atc --framework=5 \
    --model="$model_path" \
    --input_format=NCHW \
    --input_shape="images:${batchsize},3,${imgsz},${imgsz}" \
    --output_type=FP16 \
    --output="$output_name" \
    --log=error \
    --soc_version=Ascend${chip_name}


if [ $? -eq 0 ]; then
    echo ""
    echo "Conversion completed successfully!"
    echo "Output file: ${output_name}.om"

    if [ -f "${output_name}.om" ]; then
        file_size=$(ls -lh "${output_name}.om" | awk '{print $5}')
        echo "File size: $file_size"
    fi

    echo ""
    echo "You can now use the OM model with infer.py:"
    echo "  python infer.py --pth=${output_name}.om --dataset=your_dataset.yaml"

    case $task in
        "segment")
            echo "  Suggested dataset: coco.yaml"
            ;;
        "classify")
            echo "  Suggested dataset: imagenet"
            ;;
        "pose")
            echo "  Suggested dataset: coco-pose.yaml"
            ;;
        "obb")
            echo "  Suggested dataset: DOTAv1.yaml"
            ;;
        *)
            echo "  Suggested dataset: coco.yaml"
            ;;
    esac

else
    echo ""
    echo "❌ Conversion failed!"
    echo "Please check the error messages above and ensure:"
    echo "1. The ONNX model is valid"
    echo "2. ATC tool version is compatible"
    echo "3. Input shape matches the model's expected input"
    echo "4. Sufficient memory is available"
    exit 1
fi