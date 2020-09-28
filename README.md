This repository contains the code for our work **Graph Filtration Learning** which was accepted at ICML'20.

The current version will be partly refactored and additionally documented up to end of august 2020.

# Installation

In the following `<root_dir>` will be the directory in which you have chosen to do the installation.

1. Install Anaconda from [https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh](here) into `<root_dir>/anaconda3`, i.e., set the prefix accordingly in the installer. 

2. Activate Anaconda installation: `source <root_dir>/anaconda3/bin/activate`. 

3. Install pytorch from [https://pytorch.org/](here). 



. Install `pytorch-geometric` and its dependencies following the instructions on its [https://github.com/rusty1s/pytorch_geometric](gh-page).

. Install `torchph` via 

    `git clone -b 'submission_icml2020' --single-branch --depth 1 https://github.com/c-hofer/torchph.git`
    `conda develop torchph
