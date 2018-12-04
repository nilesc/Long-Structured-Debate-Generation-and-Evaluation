#!/bin/bash
cd ..

# Set up and load environment
echo ". /home/edb2129/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
. /home/edb2129/miniconda3/etc/profile.d/conda.sh
yes | conda create -n py36 python=3.6
conda activate py36

# Install scraper requirements
yes | conda install -c anaconda beautifulsoup4 
yes | conda install -c conda-forge selenium
yes | conda install -c conda-forge progressbar2
yes | conda install -c conda-forge spacy 
yes | python -m spacy download en
yes | conda install -c conda-forge langdetect

# Get repo:
git clone https://github.com/pytorch/fairseq.git
cd fairseq

# Install fairseq requirements
yes | conda install pytorch torchvision -c pytorch
yes | conda install -c anaconda cython 
while read requirement; do conda install --yes $requirement; done < requirements.txt
python setup.py build develop

cd ..
