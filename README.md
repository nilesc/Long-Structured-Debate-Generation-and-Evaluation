# debatesim
debatesim is a project aiming to use deep learning techniques to help a program learn to make meaningful arguments in response to a prompt.

# Scraping Instructions

## Install Requirements
In order to gather data, selenium must be installed. Selenium can be installed through the following command.

    conda install -c conda-forge selenium

Additional requirements:

* BeautifulSoup4

## Run Scraper

First, scrape the debates from kialo. This will take up to an hour.

    cd scraper/
    python download_debates.py

Next, we filter problematic debates. From the root directory run:

    ./scraper/filter_debates.py


# Build Training / Val / Test data

From the root run:

    python tree_builder.py

This will construct source and target data and place it in `./input_files/`


# Setup environment

Optionally, install Anaconda, NVIDIA Drivers, and CUDA. Run:

    ./setup-instance.sh

Install other dependencies and train a fairseq model on the writingPrompts dataset.

    ./setup-environment.sh
    
