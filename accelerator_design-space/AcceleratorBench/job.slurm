#!/bin/bash
#SBATCH --job-name=smallembeddings       # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G                # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1                    # number of gpus per node
#SBATCH --time=144:00:00                # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends
#SBATCH --mail-type=fail                # send email if job fails
#SBATCH --mail-user=chli@princeton.edu

module purge

module load anaconda3/2021.5
conda activate cnnbench

python get_embeddings.py
