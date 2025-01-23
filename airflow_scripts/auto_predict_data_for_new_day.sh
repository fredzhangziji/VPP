#!/bin/bash

export PATH=$PATH:/usr/local/bin
CONDA_PATH="/home/elu/software/anaconda3"
PROJECT_PATH="/home/elu/VPP/airflow_scripts"
echo "Starting script execution..."
cd "${PROJECT_PATH}"

eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
conda activate prophet_env

# Execute Python script
if [ -f "predict_data_for_new_day.py" ]; then
    python predict_data_for_new_day.py
else
    echo "Error: Python script not found"
    exit 1
fi