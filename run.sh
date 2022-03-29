#!/bin/bash

USE_N_GPU=5
MAIN_ENTRY="./main.py"
export CUDA_VISIBLE_DEVICES="3,4,5,6,7"
export OMP_NUM_THREADS=2 


echo ""
echo -e "\033[43;44m USING $USE_N_GPU GPUs TO TRAIN \033[0m"

python -m torch.distributed.run --nproc_per_node $USE_N_GPU $MAIN_ENTRY