#!/bin/bash

export PATH=$PATH:/usr/local/bin
echo -n "开始执行脚本"
path=$(cd `dirname $0`; pwd)
echo $path
cd $path
source ../active_env.sh
source $activate_path
cd $project_path
python fecth_train_predict_data.py