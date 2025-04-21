#!/bin/bash

export PATH=$PATH:/usr/local/bin
CONDA_PATH="/home/elu/software/anaconda3"
PROJECT_PATH="/home/elu/VPP/renewable_data_cralwer"
echo "Starting script execution..."
cd "${PROJECT_PATH}"

eval "$("${CONDA_PATH}/bin/conda" shell.bash hook)"
conda activate baseline_model

# Execute Python script
if [ -f "wind_power_latest_data_crawler.py" ]; then
    python wind_power_latest_data_crawler.py
else
    echo "Error: Python script not found"
    exit 1
fi