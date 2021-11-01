#!/usr/bin/env bash

export CONDA_ENV_NAME=fthe-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install numpy==1.18.5 torch==1.7.1 torchvision==0.8.2
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r requirements.txt
