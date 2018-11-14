#!/usr/bin/env bash

# Set up and load environment
yes | conda create -n py36 python=3.6
source activate py36

# Get repo:
git clone https://github.com/pytorch/fairseq.git
cd fairseq

# Get packages
yes | conda install pytorch torchvision -c pytorch
pip install Cython
pip install -r requirements.txt
python setup.py build develop