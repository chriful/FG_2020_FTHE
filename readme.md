
# FTHE:Frame-level Feature Tokenization Learning for Human Body Pose and Shape Estimation

## Introduction

This repository is the official Pytorch implementation of Frame-level Feature Tokenization Learning for Human Body Pose and Shape Estimation. The base codes are largely borrowed from [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE).

## Quick start
FTHE has been implemented and tested on Ubuntu 16.04 with Pytorch 1.7 and python 3.7.

Clone the repo:

```
git clone https://github.com/chriful/FG_2020_FTHE.git
```

Install the requirements using `pip` or `conda`:

```
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```
## Quick demo
Download the pre-trained demo FTHE and required data by below command and download SMPL layers from [here](http://smplify.is.tue.mpg.de/) (neutral). Put SMPL layers (pkl files) under `${ROOT}/data/fthe_data/`

```
source scripts/prepare_data.sh
```

Run the demo code

```
python demo.py --vid_file sample_video.mp4 --output_folder output/
```

A video overlayed with rendered meshes will be saved in `${ROOT}/output/demo_output/`.
## Training

Run the commands below to start training:

```
source scripts/prepare_training_data.sh
python train.py --cfg configs/config_w_3dpw.yaml
```

Note that the training datasets should be downloaded and prepared before running data processing script. Please see [here](https://github.com/mkocabas/VIBE/blob/master/doc/train.md) for details on how to prepare them.
## Evaluation

* Download pre-trained FTHE weights from [here].
* Run the evaluation code with a corresponding config file to reproduce the performance in the tables of our paper.

```
# dataset: 3dpw, mpii3d

# Table I
python evaluate.py --dataset 3dpw --cfg ./configs/config_w_3dpw_model.yaml
# Table II
python evaluate.py --dataset 3dpw --cfg ./configs/config_wo_3dpw_model.yaml
```
