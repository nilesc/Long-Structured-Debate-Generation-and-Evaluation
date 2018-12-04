#!/bin/bash

# Update apt-get repos
sudo apt-get update

# Get bunzip2
yes | sudo apt-get install bzip2

# Optional utilities
sudo apt-get install tmux
sudo apt-get install htop

# Get conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
