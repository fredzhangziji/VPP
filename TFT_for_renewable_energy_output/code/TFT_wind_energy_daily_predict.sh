#!/bin/bash

export PATH=$PATH:/usr/local/bin
CONDA_PATH="/home/elu/software/anaconda3"
PROJECT_PATH="/home/elu/VPP/TFT_for_renewable_energy_output/code"
echo "Starting wind energy TFT daily predict script execution..."
cd "${PROJECT_PATH}"

eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
conda activate TFT

# Execute Python script
if [ -f "predict.py" ]; then
    python predict.py
else
    echo "Error: Python script not found"
    exit 1
fi