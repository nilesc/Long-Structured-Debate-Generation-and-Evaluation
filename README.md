# debatesim

debatesim is a project aiming to use deep learning techniques to help a program learn to make meaningful arguments in response to a prompt. It is based on the [fairseq](https://github.com/pytorch/fairseq) project.

## Scraping Instructions

### Install Requirements
In order to gather data, selenium must be installed. Selenium can be installed through the following command.

    conda install -c conda-forge selenium

Additional requirements:

  * BeautifulSoup4

### Run Scraper

First, crawl kialo, downloading all debates through their export feature. This will take up to an hour.

    cd scraper/
    python download_debates.py

Next, we filter problematic debates. These are debates that are either formatted in a way that makes them hard to parse, or in a language other than English.

    python filter_debates.py

This will copy all appropriate debates to a folder named `./filtered_debates/`.

## Build Training / Val / Test data

From the root run:

    python tree_builder.py

This will construct source and target data and place it in `./input_files/`. The tree_builder script gives several options for how to build the tree, including whether to include only Pro aguments, only Con arguments, or both, and whether or not to augment the data with sub-trees.

## Setup environment

Optionally, install Anaconda, NVIDIA Drivers, and CUDA. Run:

    ./setup-instance.sh

Install other dependencies and train a fairseq model on the writingPrompts dataset.

    ./setup-environment.sh
