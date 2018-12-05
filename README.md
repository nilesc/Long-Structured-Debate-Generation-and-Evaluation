# debatesim

debatesim is a project aiming to use deep learning techniques to help a program learn to make meaningful arguments in response to a prompt. It is based on the [fairseq](https://github.com/pytorch/fairseq) project. It does this by training on information gathered from the debate website Kialo.

We collected a data set from Kialo, an online debateplatform, and trained the model developed by Fan et al. (2018) to generate arguments in favor or against given de-bate prompts. Kialo debates come in a structured tree format, based on core prompts such as “An artificial general intelligence (AGI) should be created.” Users submit pros and cons, which are then approved by a moderator. New users can then respond to those pros and cons as though they themselves were prompts, forming a debate tree. We explored different ways to pair create prompts and responses from our debate tree structures, creating a rich dataset from comparatively few debates, and experimented with other ways of improving the efficacy of the Fan et al. (2018) model for our task. We also explored the ease of adapting the techniques and code-base of the previous study.

Chosen results:

Prompt: | Generated Response
---|---
Buddhism has dogma . | Buddhism is the only scientific method of making the choice , but it is impossible to say in the best interest in society for many in other religions .
TV unites people . | TV creates a dangerous and bad lifestyle . Seeing the quality of life to some extent , you can help to make the life easier .
Welfare can reduce crime . | The social benefits of the offender should be taken into account and can not be done .
Morality is \<unk\> | This is a bit of a spiritual perspective . We are not talking about things that can not be used for a good reason . There 's a reason for a person to be \<unk\> , and we do n't know whether we are \<unk\> and \<unk\> .

Note: \<unk\> represents words that do not appear frequently enough to be in our vocabulary.

## Setup Instructions

### Getting our Code

Before anything else, ensure that you have the code from this repository available on the machine you intend to run this.
The code for this section is intended for setting up a fresh VM that has at least one GPU and is running Ubuntu 16.04 LTS. The fairseq model will not train unless you are using a machine with at least one available GPU.

In order to do this, update apt-get:

    $ sudo apt-get update

Next, use apt-get to install git:

    $ sudo apt-get install git

Now, clone our repository with git:

    git clone https://github.com/nnc2117/debatesim.git

### Setup Instance

The instance we used was a Google Cloud Compute "Deep Learning VM" deployed from this [link](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning?angularJsUrl=%2Fmarketplace%2Fdetails%2Fclick-to-deploy-images%2Fdeeplearning&authuser=2), with a second GPU added. This resulted in following settings:

| Name   | Setting                         |
|--------|---------------------------------|
| OS     | AMD 4.9.0-8                     |
| CPU    | 2 vCPUs                         |
| Memory | 13 GB                           |
| GPU    | 2 x NVIDIA Tesla K80            |
| HD     | 100 GB Standard Persistent Disk |

The code for this section can be executed with the following script:

    $ bash setup-instance.sh

If you would just like to get our code up and running as quickly as possible, just run the above script and skip to scraping instructions below.

If you would instead like a step-by-step walkthrough of what the code is doing, follow along below.

First, get bunzip2:

    $ sudo apt-get install bzip2

Next, install optional utilities to assist with model training runs.

    $ sudo apt-get install tmux
    $ sudo apt-get install htop

Finally, get conda to manage your environment:

    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh

Follow the install instructions and restart your VM.

## Scraping Instructions

### Install Requirements

The code for this subsection can be run with:

    $ bash setup-environment.sh

First things first, setup a conda environment:

    $ echo ". /home/edb2129/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    $ . /home/edb2129/miniconda3/etc/profile.d/conda.sh
    $ conda create -n py36 python=3.6
    $ conda activate py36

In order to gather data, selenium must be installed. Selenium can be installed through the following command. Note that this requires conda to be installed. If you have set up your instance according to the instructions above, this should be taken care of already.

Additional requirements:

  * BeautifulSoup4

        conda install -c anaconda beautifulsoup4

  * Selenium

        conda install -c conda-forge selenium

  * Progressbar

        conda install -c anaconda progressbar2

  * spacy (with english language model)

        conda install -c conda-forge spacy
        python -m spacy download en

  * langdetect

        conda install -c conda-forge langdetect

### Run Crawler

(If you'd rather just download our dataset without crawling it, it can be found [here]())

First, crawl Kialo, downloading all debates through their export feature. This will take up to an hour. This will download all available debates onto your system to use as a training corpus.

    cd data_processing/
    python crawl_debates.py

This will place debates into your download directory. From there, put them into a new directory called `./data/discussions/`.

Next, we filter problematic debates. These are debates that are either formatted in a way that makes them hard to parse, or in a language other than English.

    python filter_debates.py

This will copy all appropriate debates to a folder named `./data/filtered_discussions/`.

## Build Training / Val / Test data

    python tree_builder.py

This will construct source and target data and place it in `./data/input_files/`. For each debate, all traversals of the tree corresponding to coherent arguments will be added to a `target` file, and the corresponding debate prompt will be added to a `source` file. This script gives several options for how to build the tree, including whether to include only Pro aguments, only Con arguments, or both, and whether or not to augment the data with sub-trees. By default, all traversals from the root involving only positive children are included. Note that we also include traversals that do not end at a leaf node. In this way, we get substrings of arguments that are themselves coherent arguments.

## Setup model

This section involves cloning a version of the fairseq project, setting up an environment in which it will run without a hitch, and running all aspects of the story generation task on the reddit [writingPrompts](https://www.google.com/search?q=reddit+writingprompts&rlz=1C5MACD_enUS504US504&oq=reddit+writingprompts&aqs=chrome.0.0l6.3305j0j1&sourceid=chrome&ie=UTF-8) dataset. You can skip step 3 if you want to get straight to generating talking points using the Kialo dataset.

### 1. Install dependencies
The code for this subsection can be run with:

    $ bash setup-environment.sh

First, get the fairseq repository and open it.

    $ git clone https://github.com/pytorch/fairseq.git
    $ cd fairseq

Finally, install all the requirements.

    $ conda install pytorch torchvision -c pytorch
    $ conda install -c anaconda cython
    $ while read requirement; do conda install --yes $requirement; done < requirements.txt
    $ python setup.py build develop

### 2. Train a fairseq model on the writingPrompts dataset
(Skip this section if you want to get straight to using the Kialo dataset)

The code for this subsection can be run with:

    $ bash train-model.sh

First, download all the writingPrompts data into examples/stories

    $ cd examples/stories
    $ curl https://s3.amazonaws.com/fairseq-py/data/writingPrompts.tar.gz | tar xvzf -

Now, preprocess the data.

    $ cd ../../..
    $ TEXT=examples/stories/writingPrompts
    $ python preprocess.py --source-lang wp_source --target-lang wp_target \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/writingPrompts --padding-factor 1 --thresholdtgt 10 \
        --thresholdsrc 10 --workers 8

Next, download relevant checkpoints for the model and place them in the data-bin folder.

    $ curl https://s3.amazonaws.com/fairseq-py/models/stories_checkpoint.tar.bz2 | tar xvjf - -C data-bin

This line will train a fusion model using the downloaded pretrained checkpoint:

    $ python train.py data-bin/writingPrompts -a fconv_self_att_wp --lr 0.25 \
        --clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau \
        --decoder-attention True --encoder-attention False --criterion \
        label_smoothed_cross_entropy --weight-decay .0000001 --label-smoothing 0 \
        --source-lang wp_source --target-lang wp_target --gated-attention True \
        --self-attention True --project-input True --pretrained True \
        --pretrained-checkpoint data-bin/models/pretrained_checkpoint.pt
        # --distributed-world-size 8  # Add this line to run in with multiple processes

In order to train a non-fusion model, replace:

        --pretrained True \
        --pretrained-checkpoint data-bin/models/pretrained_checkpoint.pt

With:

        --pretrained False

Finally, use the downloaded fusion checkpoint to perform the generation task.

    $ python generate.py data-bin/writingPrompts --path \
        data-bin/models/fusion_checkpoint.pt --batch-size 32 --beam 1 \
        --sampling --sampling-topk 10 --sampling-temperature 0.8 --nbest 1 \
        --model-overrides \
        "{'pretrained_checkpoint':'data-bin/models/pretrained_checkpoint.pt'}"

## Run model on Kialo dataset
This section modifies subsection 2 from the previous section in order to generate debate responses using the Kialo training set. This training set must be created using the tree builder.

Now, perform the full preprocessing of the data.

    $ cd ..
    $ mv data/input_files fairseq/examples/kialo
    $ cd fairseq
    $ TEXT=examples/kialo
    $ python preprocess.py --source-lang kialo_source --target-lang kialo_target \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/kialo --padding-factor 1 --thresholdtgt 10 \
        --thresholdsrc 10 --workers 8

To train a non-fusion model, use the following line:

    $ python train.py data-bin/kialo -a fconv_self_att_wp --lr 0.25 \
        --clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau \
        --decoder-attention True --encoder-attention False --criterion \
        label_smoothed_cross_entropy --weight-decay .0000001 --label-smoothing 0 \
        --source-lang kialo_source --target-lang kialo_target --gated-attention True \
        --self-attention True --project-input True --pretrained False \
        # --distributed-world-size 8  # Add this line to run with multiple processes

To train a fusion model, empty your `checkpoints` file and make sure to save a checkpoint separately:

    $ mkdir data-bin/models
    $ mv checkpoints/checkpoint_best.pt data-bin/models
    $ rm checkpoints/*

Then run the `train.py` command above with:

        --pretrained True --pretrained-checkpoint data-bin/models/checkpoint_best.pt

Instead of `--pretrained False`.

Finally, perform the generation task:

    $ python generate.py data-bin/kialo --path \
        checkpoints/checkpoint_best.pt --batch-size 32 --beam 1 \
        --sampling --sampling-topk 10 --sampling-temperature 0.8 --nbest 1
