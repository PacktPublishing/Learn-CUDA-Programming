#/bin/bash

CODE_PATH="RN50v1.5"
DATASET_PATH="/raid/datasets/imagenet/raw-data/"
OUTPUT_NAME="resnet50_pyt"

# default profile
docker run --rm -ti --runtime=nvidia \
    -v $(pwd)/${CODE_PATH}:/workspace \
    -v ${DATASET_PATH}:/imagenet \
    nvcr.io/nvidia/pytorch:19.08-py3 \
       nsys profile -t cuda,nvtx,cudnn,cublas -o ${OUTPUT_NAME} -f true -w true -y 60 -d 20 \
       python /workspace/main.py --arch resnet50 -b 64 --fp16 /imagenet
