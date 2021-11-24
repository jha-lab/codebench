# Simulator
Simulator of the accelerator

1. Parameters (area, dynamic power, leakage power, etc.) are defined in `defines.py`.
2. Dissescting CNN frozen graph `xxx.pb` into blocks in `pb2blocks.py`.
3. Blocks are defined in `Blocks.py`.
4. Hardware modules are described and defined in `Hardware.py`.
5. To run **SPRING**, use `run.py`. See below for details.

### Running SPRING on Nobel & Adroit at Princeton
#### Nobel
1. Prepare at least 3GB of storage on your home drive.
2. Install `tensorflow` on your home drive. Follow the instructoin here: <https://researchcomputing.princeton.edu/tensorflow>
3. Use command: `$ python run.py <file name of the frozen graph>` e.g.: `$ python run.py ./files/nets_inference/mobilnet_v2.pb`

#### Adroit
1. Apply for an account on Adroit on <https://forms.rc.princeton.edu/registration/>
2. Install `tensorflow` with the instructions here: <https://researchcomputing.princeton.edu/tensorflow>
3. Submit job though the `Slurm Scheduler`. Follow the example and instructions here: <https://researchcomputing.princeton.edu/tensorflow>
4. Example Slurm script for **SPRING**
```script
#!/bin/bash
#SBATCH --job-name=mobilnet_v2   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4G per cpu-core is default)
# SBATCH --gres=gpu:tesla_v100:1  # USE THIS LINE ON ADROIT BY REMOVING SPACE BEFORE SBATCH
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=chli@princeton.edu

module purge
module load anaconda3
conda activate tf2-gpu

python run.py ./files/nets_inference/nasnet_mobile.pb
```
5. Better to comment the lines **157-160** and **189-192** in `Accelerator.py` to prevent the `slurm.xxxxxx.out` log file from draining your quota and causing a termination to your job. Those are the print functions for cycle-wise information during processing, not necessary for the simulation results. 
6. Use command: `$ sbatch job.slurm`	to submit the job
7. Use command: `$ squeue -u <NetID>`	to check the status of your job
8. For more about slurm, go to <https://researchcomputing.princeton.edu/slurm>
