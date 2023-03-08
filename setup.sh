#!/bin/bash

conda create -n cutalk python=3.10
conda activate cutalk
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y scipy
#pip install wxpython
conda install -y -c conda-forge wxpython
conda install -y matplotlib
