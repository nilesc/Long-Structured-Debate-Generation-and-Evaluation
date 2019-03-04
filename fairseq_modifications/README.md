In order to support multiturn debate generation, wherein our language model
generates new responses based on previously generated ones, we made
modifications to some of the files in the Facebook fairseq [Github
repository](https://github.com/pytorch/fairseq).

These files are copied here. Simply replace the relevant files in the fairseq
directory with these in order to enable multiturn:

    $ FAIRSEQ_PATH=<insert path to fairseq repository here>
    $ rm $FAIRSEQ_PATH/fairseq/options.py
    $ rm $FAIRSEQ_PATH/generate.py
    $ cp options.py $FAIRSEQ_PATH/fairseq
    $ cp generate.py $FAIRSEQ_PATH

Now, you should be able to generate multiturn debates:

    $ cd $FAIRSEQ_PATH
    $ touch examples/kialo/output.kialo_source
    $ python generate.py data-bin/kialo --path \
        checkpoints/fusion_checkpoint.pt --batch-size 32 --beam 1 \
        --sampling --srcdict data-bin/kialo/dict.kialo_source.txt \
        --tgtdict data-bin/kialo/dict.kialo_target.txt --sampling-topk 10 \
        --sampling-temperature 0.8 --nbest 1 --unkpen 2.0 \
        --source-lang kialo_source --target-lang kialo_target --multiturn \
        --multiturnpref $TEXT/multiturn --outputpref $TEXT/output --testpref \
        $TEXT/test --destdir data-bin/kialo --padding-factor 1 \
        --thresholdtgt 10     --thresholdsrc 10 --workers 1 --model-overrides \
        "{'pretrained_checkpoint':'data-bin/models/checkpoint_best.pt'}"

The results will be stored in `examples/kialo/output.kialo_source`
