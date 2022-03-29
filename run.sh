#!/bin/bash

USE_N_GPU=4
MAIN_ENTRY="./main.py"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export OMP_NUM_THREADS=2 


echo ""
echo -e "\033[43;44m USING $USE_N_GPU GPUs TO TRAIN \033[0m"


# -f flag for specify configuration file location
# -s flag for specify whether to save a copy of current default configuration
python -m torch.distributed.run --nproc_per_node $USE_N_GPU $MAIN_ENTRY -s True -f config.json