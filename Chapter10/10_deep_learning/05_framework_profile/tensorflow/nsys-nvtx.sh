#/bin/bash

CODE_PATH="RN50v1.5"
DATASET_PATH="/raid/datasets/imagenet/tfrecord"
OUTPUT_NAME="resnet50_tf"

# default profile
docker run --rm -ti --runtime=nvidia \
    -v $(pwd):/result \
    -v $(pwd)/${CODE_PATH}:/workspace \
    -v ${DATASET_PATH}:/imagenet \
    nvcr.io/nvidia/tensorflow:19.08-py3 \
        nsys profile -t cuda,nvtx,cudnn,cublas -o ${OUTPUT_NAME} -f true -w true -y 40 -d 20 \
            python /workspace/main.py --data_dir=/imagenet --mode=training_benchmark --warmup_steps 200 \
                --num_iter 500 --iter_unit batch --results_dir=results --batch_size 64
