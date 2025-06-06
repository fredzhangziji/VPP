#!/bin/bash

export PATH=$PATH:/usr/local/bin
CONDA_PATH="/home/elu/anaconda3/"
PROJECT_PATH="/home/elu/VPP/Baseline/Wind_Power_Output_Baseline/code"
echo "Starting wind energy baseline daily predict script execution..."
cd "${PROJECT_PATH}"

eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
conda activate baseline

# Execute Python script
if [ -f "future_predict.py" ]; then
    python future_predict.py
else
    echo "Error: Python script not found"
    exit 1
fi