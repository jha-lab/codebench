#!/bin/sh

# Script to install required packages in conda for GPU setup
# Author : Shikhar Tuli

module load anaconda3
conda create --name cnnbench tensorflow-gpu

conda activate cnnbench

conda install -c powerai tensorflow-datasets