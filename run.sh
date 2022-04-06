#!/bin/bash

USE_N_GPU=4
MAIN_ENTRY="./main.py"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=2

CONFIG_FILE=${1:-"config.json"}

echo -e "\033[104;097m USING $USE_N_GPU GPUs TO TRAIN \033[0m"

echo -e "\033[104;097m LOADING \033[090;102m$CONFIG_FILE \033[0m \033[0m"

# -f flag for specify configuration file location
# -s flag for specify whether to save a copy of current default configuration
python -m torch.distributed.run --nproc_per_node $USE_N_GPU $MAIN_ENTRY -s True -f "$CONFIG_FILE"