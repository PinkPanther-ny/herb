#!/bin/bash

MAIN_ENTRY="./main.py"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS=2

CONFIG_FILE=${1:-"default.json"}
USE_N_GPU=${2:-4}

echo -e "\033[104;097m USING $USE_N_GPU GPUs TO TRAIN \033[0m"

echo -e "\033[104;097m LOADING \033[090;102m$CONFIG_FILE \033[0m \033[0m"

# -f flag for specify configuration file location
python -m torch.distributed.run --nproc_per_node $USE_N_GPU $MAIN_ENTRY -f "$CONFIG_FILE"
