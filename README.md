[![arXiv](https://img.shields.io/badge/arXiv-2606.28486-b31b1b.svg)](https://arxiv.org/abs/2606.28486)

# Spectral phase transitions and trainability in neural network learning dynamics

Data and code release for [arXiv:2606.28486](https://arxiv.org/abs/2606.28486)

## Repository Structure

```
.
├── data                            <- Trained models used in the paper
├── collected_results               <- Directory for output of the workflow and notebooks
│   └── run_collected
├── env                             <- Environment files
├── libs                            <- Submodules including figure styles
├── LICENSE
├── README.md
└── src
    ├── configs                     <- json for Teacher-Student case
    ├── experiment_2                <- json for UTKFace dataset
    ├── models                      <- architectures to be trained
    ├── notebooks                   <- notebooks to run codes to produce figures
    └── training                    <- files to run the trainings
    
```

## Setup

1. Clone this repository including submodules (or download its Zenodo release and ```unzip``` it) and ```cd``` into it:

```
git clone --recurse-submodules https://github.com/f-damico/BBP_in_SGD.git
cd BBP_in_SGD
```

3. Set up the environment

You can create and activate a new conda environment with the following command:

```
conda env create -f env/environment_bbp.yml
conda activate bbp
```

## Running the workflow

### Reproducing the figures

The figures in the article can be reproduced by the jupyter notebook in ```src/notebooks```.

The figures in the paper are in folder ```figures/```.

The dataset should be downloaded and placed in the ```dataset/``` directory by running ```./download_age_dataset.py```

### Regenerating the trained models

The models saved in ```data/``` are generated with a PBS scheduler in ```run_experiment_array.pbs```. From this file it is straightforward to 
run experiments following the specifics of the user's cluster.

## Output

Output plots from the jupyter notebook are placed in the ```collected_results/figures``` directory, while the trained models are saved in the ```collected_results/models``` directory.

## Reproducibility

The original data used in the paper are hosted on Zenodo.

To reproduce the original figures, download ```data.tar.gz``` and ```raw_data.tar.gz``` from Zenodo, then place it in the ```data/``` directory and extract it with the following command:

```
mv data.tar.gz data/
tar -xvzf data/data.tar.gz data/
mv raw_data.tar.gz data/
tar -xvzf data/raw_data.tar.gz data/
```
