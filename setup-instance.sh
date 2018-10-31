# Download requisite packages
yes | sudo apt-get install git
git clone https://github.com/python/cpython.git
yes | sudo apt-get update
yes | sudo apt-get upgrade
yes | sudo apt-get dist-upgrade
yes | sudo apt-get install build-essential python-dev python-setuptools python-pip python-smbus
yes | sudo apt-get install libncursesw5-dev libgdbm-dev libc6-dev
yes | sudo apt-get install zlib1g-dev libsqlite3-dev tk-dev
yes | sudo apt-get install libssl-dev openssl
yes | sudo apt-get install libffi-dev
cd cpython
./configure
make
sudo make altinstall
cd ~
export PATH="$PATH:/usr/local/lib"
echo 'export PATH="$PATH:/usr/local/lib"' >> .bashrc
python3.8 -m pip install torch torchvision

# Get repo:
git clone https://github.com/edbltn/dl-text-generation.git

# Get data:
cd dl-text-generation/examples/stories
curl https://s3.amazonaws.com/fairseq-py/data/writingPrompts.tar.gz | tar xvzf -

# Preprocess data:
cd writingPrompts
python3.8 ../preprocess.py
cd ../../..
TEXT=examples/stories/writingPrompts
python3.8 preprocess.py --source-lang wp_source --target-lang wp_target \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/writingPrompts --padding-factor 1 --thresholdtgt 10 \
    --thresholdsrc 10

# Load checkpoints
curl https://https://s3.amazonaws.com/fairseq-py/models/stories_checkpoint.tar.bz2 | tar xvjf - -C data-bin

# Train model:
python3.8 train.py data-bin/writingPrompts -a fconv_self_att_wp --lr 0.25 \
--clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau \
--decoder-attention True --encoder-attention False --criterion \
label_smoothed_cross_entropy --weight-decay .0000001 --label-smoothing 0
\
--source-lang wp_source --target-lang wp_target --gated-attention True \
--self-attention True --project-input True --pretrained True \
--pretrained-checkpoint data-bin/models/pretrained_checkpoint.pt

# Generate:
python3.8 generate.py data-bin/writingPrompts --path \
    data-bin/models/fusion_checkpoint.pt --batch-size 32 --beam 1 \
    --sampling --sampling-topk 10 --sampling-temperature 0.8 --nbest 1 \
    --model-overrides \
    "{'pretrained_checkpoint':'data-bin/models/pretrained_checkpoint.pt'}"

