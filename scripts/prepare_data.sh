#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://drive.google.com/file/d/1wwTJYPWz61e3tYs45glcBpa47gckHGQQ/view?usp=sharing"
unzip fthe_data.zip
rm fthe_data.zip
cd ..
mv data/fthe_data/sample_video.mp4 .
mkdir -p $HOME/.torch/models/
mv data/fthe_data/yolov3.weights $HOME/.torch/models/
