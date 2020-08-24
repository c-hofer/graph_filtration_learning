This repository contains the code for our work **Graph Filtration Learning** which was accepted at ICML'20.

The current version will be partly refactored and additionally documented up to end of august 2020.

# Installation

In the following `<root_dir>` will be the directory in which you have chosen to do the installation.

1. Install Anaconda from [here](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh) into `<root_dir>/anaconda3`, i.e., set the prefix accordingly in the installer. 

2. Activate Anaconda installation: 

    ```
    source <root_dir>/anaconda3/bin/activate
    ```


3. Install pytorch via conda

    ```
    conda install pytorch=1.4.0 torchvision cudatoolkit=<your_cuda_version> -c pytorch
    ```



4. Install `pytorch-geometric` and its dependencies following the instructions on its [gh-page](https://github.com/rusty1s/pytorch_geometric).

5. Install `torchph` via 

    ```
    cd <root_dir>
    git clone -b 'submission_icml2020' --single-branch --depth 1 https://github.com/c-hofer/torchph.git
    conda develop torchph
    ```
6. Clone this repository into `<root_dir>`. 

# Application

1. Generate the experiment configurations you want using the `write_exp_cfgs_file.ipynb` notebook. It is assumed that the notebook server is started in `<root_dir>/graph_filtration_learning`.

2. Use the `train.py` script to run the experiments, e.g., 
    ```
    python train.py --cfg_file <my_cfg.json> --output_dir results --devices 0,1 --max_process_on_device 2 
    ```
    to use cuda device 0 and 1 with at most 2 experiments on each. 

    Each experiment gets a unique id and its output is written to `results` as a pickle file. Additionally for each CV run the corresponding trained model is dumped. 

3. The notebook `results.ipynb` contains some code to browse the results. 


