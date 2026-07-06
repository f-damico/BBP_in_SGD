[![arXiv](https://img.shields.io/badge/arXiv-2606.28486-b31b1b.svg)](https://arxiv.org/abs/2606.28486)

# Spectral phase transitions and trainability in neural network learning dynamics

Data and code release for [arXiv:2606.28486](https://arxiv.org/abs/2606.28486)

## Repository Structure

```
.
├── data                            <- Trained models used in the paper
├── collected_results               <- Directory for output of the workflow and notebooks
│   ├── figures
│   └── models
├── env                             <- Environment files
├── libs                            <- Submodules including figure styles
├── LICENSE
├── README.md
└── src
    ├── bin                         <- Python source code for generating trained models
    ├── notebooks                   <- Jupyter notebooks for reproducing the figures in the paper
    └── scripts                     <- Bash scripts for running the workflow
```

## Requirements

- Numpy
- Scipy
- Matplotlib
- Jupyter
- [Other dependencies]

## Setup

1. Install the dependencies above.

2. Clone this repository including submodules (or download its Zenodo release and ```unzip``` it) and ```cd``` into it:

```
git clone --recurse-submodules https://github.com/f-damico/BBP_in_SGD.git
cd BBP_in_SGD
```

3. Set up the environment

For conda users, you can create a new environment with the following command:

```
conda env create -f env/environment.yml -n [env_name]
```

Then, activate the environment:

```
conda activate [env_name]
```

For pip users, you can install the required packages with the following command:

```
python -m pip install -r env/requirements.txt
```

## Running the workflow

### Reproducing the figures

The figures in the article can be reproduced by the jupyter notebook in ```src/notebooks```.

The dataset should be downloaded and placed in the ```data/``` directory as described in the Reproducibility section below.

### Regenerating the trained models

The models saved in ```data/``` are generated from the training source code in ```src/bin/linear_bbp3.py```.

## Output

Output plots from the jupyter notebook are placed in the ```collected_results/figures``` directory, while the trained models are saved in the ```collected_results/models``` directory.

## Reproducibility

The original data used in the paper will be hosted on Zenodo and linked here.

To reproduce the original figures, download ```data.tar.gz``` from Zenodo, then place it in the ```data/``` directory and extract it with the following command:

```
mv data.tar.gz data/
tar -xvzf data/data.tar.gz data/
```

## How I plan to structure the code
- A folder for each experiment (for now, first experiment will be to repeat Fig. 2 of the paper), that accept as input a general architecture + training
- A folder for all architectures, with one separate file for any of them. The architecture is defined but the widths and depths are passed as an argument, with the possibility to select $\sigma_W$
- A folder for training setups, where are passed instructions about training hyperparameters as learning rate, number of epochs, batch size

## First experiment
I will repeat the setup of a DNN with fixed input and output layers equal to the Teacher 
