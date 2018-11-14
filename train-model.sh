# Get data:
cd examples/stories
curl https://s3.amazonaws.com/fairseq-py/data/writingPrompts.tar.gz | tar xvzf -

# Preprocess data:
cd writingPrompts
echo 'data = ["train", "test", "valid"]
for name in data:
  with open(name + ".wp_target") as f:
    stories = f.readlines()
  stories = [" ".join(i.split()[0:1000]) for i in stories]
  with open(name + ".wp_target", "w") as o:
    for line in stories:
      o.write(line.strip() + "\n")' >> preprocess.py
python preprocess.py
cd ../../..

TEXT=examples/stories/writingPrompts
python preprocess.py --source-lang wp_source --target-lang wp_target \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/writingPrompts --padding-factor 1 --thresholdtgt 10 \
    --thresholdsrc 10 --workers 8

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
# --distributed-world-size 8  # Add this line to run in with multiple processes


# Generate:
python generate.py data-bin/writingPrompts --path \
    data-bin/models/fusion_checkpoint.pt --batch-size 32 --beam 1 \
    --sampling --sampling-topk 10 --sampling-temperature 0.8 --nbest 1 \
    --model-overrides \
    "{'pretrained_checkpoint':'data-bin/models/pretrained_checkpoint.pt'}"
