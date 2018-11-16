#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
# The 16.04 installer works with 16.10.
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1amd64.deb
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repose/ubuntu1604/x86_64/7fa2af80.pub
    # If the above line throws an error, use the following instead
    # curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub > 7fa2af80.pub
    # sudo apt-key add *.pub
    apt-get update
    apt-get install cuda-9-0 -y
fi
nvidia-smi -pm 1
