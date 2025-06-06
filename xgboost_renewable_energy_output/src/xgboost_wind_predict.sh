#!/bin/bash

export PATH=$PATH:/usr/local/bin
CONDA_PATH="/home/elu/anaconda3"
PROJECT_PATH="/home/elu/VPP/xgboost_renewable_energy_output/src"
echo "Starting wind energy xgboost daily predict script execution..."
cd "${PROJECT_PATH}"

eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
conda activate xgboost

# Execute Python script
if [ -f "predict.py" ]; then
    python predict.py
else
    echo "Error: Python script not found"
    exit 1
fi