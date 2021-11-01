#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv fthe-env
echo "Activating virtual environment"

source $PWD/fthe-env/bin/activate

$PWD/fthe-env/bin/pip install numpy==1.18.5 torch==1.7.1 torchvision==0.8.2
$PWD/fthe-env/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/fthe-env/bin/pip install -r requirements.txt
