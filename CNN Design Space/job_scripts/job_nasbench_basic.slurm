#!/bin/bash

#SBATCH --job-name=nasbench_multi           # create a short name for your job
#SBATCH --nodes=1                   		# node count
#SBATCH --ntasks=2                  		# total number of tasks across all nodes
#SBATCH --cpus-per-task=2                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:tesla_k40c:2             # number of gpus per node
#SBATCH --time=01:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate cnnbench

python generate_graphs_script.py --max_vertices 2 --output_file '../results/vertices_2/generated_graphs.json'

for i in {0..1}
do
	python run_evaluation_script.py --worker_id $i --total_workers 2 --module_vertices 2 \
	--models_file '../results/vertices_2/generated_graphs.json' \
	--output_dir '../results/vertices_2/evaluation' &
done

wait

python cleanup_script.py '../results/vertices_2/evaluation'
