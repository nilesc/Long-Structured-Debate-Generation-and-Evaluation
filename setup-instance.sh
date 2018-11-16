#!/usr/bin/env bash

# Get conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Update apt-get repos
sudo apt-get update

# Get bunzip2
yes | sudo apt-get install bzip2

# Get CUDA
bash setup-cuda.sh

# Add CUDA to environment (should be added to in ~/.bashrc)
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Optional utilities
sudo apt-get install tmux
sudo apt-get install htop

