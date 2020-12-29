#!/bin/bash

#SBATCH --job-name=nasbench_multi           # create a short name for your job
#SBATCH --nodes=$numNodes                   # node count
#SBATCH --ntasks=$numTasks                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=$cluster_gpu                 # number of gpus per node
#SBATCH --time=01:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate acc-nn

python generate_graphs_script.py --max_vertices $numVertices

for i in {0..($numTasks-1)}
do
	srun -N 1 -n 1 python run_evaluation_script.py --worker_id i --total_workers $numTasks --module_vertices $numVertices &
done

wait

python cleanup_script.py