#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip fthe_data.zip
rm fthe_data.zip
cd ..
mv data/fthe_data/sample_video.mp4 .
mkdir -p $HOME/.torch/models/
mv data/fthe_data/yolov3.weights $HOME/.torch/models/
