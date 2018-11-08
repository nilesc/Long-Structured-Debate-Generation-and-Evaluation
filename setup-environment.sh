#!/usr/bin/env bash

# Set up and load environment
yes | conda create -n py36 python=3.6
source activate py36

# Get repo:
git clone https://github.com/pytorch/fairseq.git
cd dl-text-generation

# Get packages
yes | conda install pytorch torchvision -c pytorch
pip install Cython
pip install -r requirements.txt

# Get data:
cd examples/stories
curl https://s3.amazonaws.com/fairseq-py/data/writingPrompts.tar.gz | tar xvzf -

# Preprocess data:
cd writingPrompts
python ../preprocess.py
cd ../../..
TEXT=examples/stories/writingPrompts
python preprocess.py --source-lang wp_source --target-lang wp_target \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/writingPrompts --padding-factor 1 --thresholdtgt 10 \
    --thresholdsrc 10

# Load checkpoints
curl https://s3.amazonaws.com/fairseq-py/models/stories_checkpoint.tar.bz2 | tar xvjf - -C data-bin

# Train model:
python train.py data-bin/writingPrompts -a fconv_self_att_wp --lr 0.25 \
--clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau \
--decoder-attention True --encoder-attention False --criterion \
label_smoothed_cross_entropy --weight-decay .0000001 --label-smoothing 0 \
--source-lang wp_source --target-lang wp_target --gated-attention True \
--self-attention True --project-input True --pretrained True \
--pretrained-checkpoint data-bin/models/pretrained_checkpoint.pt

# Generate:
python generate.py data-bin/writingPrompts --path \
    data-bin/models/fusion_checkpoint.pt --batch-size 32 --beam 1 \
    --sampling --sampling-topk 10 --sampling-temperature 0.8 --nbest 1 \
    --model-overrides \
    "{'pretrained_checkpoint':'data-bin/models/pretrained_checkpoint.pt'}"
