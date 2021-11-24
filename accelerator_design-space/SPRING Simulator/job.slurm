#!/bin/bash
#SBATCH --job-name=inception_v3  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4G per cpu-core is default)
# SBATCH --gres=gpu:tesla_v100:1  # USE THIS LINE ON ADROIT BY REMOVING SPACE BEFORE SBATCH
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=chli@princeton.edu

module purge
module load anaconda3
conda activate tf2-gpu

python run.py ./files/nets_inference/inception_v3.pb
