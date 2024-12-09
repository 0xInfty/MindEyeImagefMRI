#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.

# set -e

# conda create -n mindeye python=3.10.8 -y
# conda activate mindeye

conda install numpy matplotlib tqdm scikit-image jupyterlab -y
# conda install numpy==1.26.4 matplotlib==3.9.2 tqdm==4.66.5 scikit-image==0.23.2 jupyterlab==4.2.5 -y
conda install -c conda-forge accelerate -y
# conda install -c conda-forge accelerate==0.34.2 -y

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install clip-retrieval webdataset clip pandas matplotlib ftfy regex kornia umap-learn
# pip install huggingface_hub==0.24.6 clip-retrieval==2.44.0 webdataset==0.2.100 clip==0.2.0 pandas==2.2.2 matplotlib==3.9.2 ftfy==6.2.3 regex==2024.9.11 kornia==0.7.3 umap-learn==0.5.6
pip install dalle2-pytorch
# pip install dalle2-pytorch==1.15.6

pip install torchvision==0.15.2 torch==2.0.1
pip install diffusers==0.13.0

pip install info-nce-pytorch==0.1.0
pip install pytorch-msssim
# pip install pytorch-msssim==1.0.0

# pip install pydantic==2.9.1
pip install pyvdirs