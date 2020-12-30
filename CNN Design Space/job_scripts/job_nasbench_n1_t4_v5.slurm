#!/bin/bash
#SBATCH --job-name=nasbench_multi           # create a short name for your job
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=4                # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:tesla_v100:+4               # number of gpus per node
#SBATCH --time=10:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate acc-nn

python generate_graphs_script.py --max_vertices 5 --output_file '../results/vertices_5/generated_graphs_vertices_5.json'

for i in {0..3}
do
  python run_evaluation_script.py --worker_id $i --total_workers 4 --module_vertices 5   --models_file '../results/vertices_5/generated_graphs_vertices_5.json'   --output_dir '../results/vertices_5/evaluation' &
done

wait

python cleanup_script.py '../results/vertices_5/evaluation'
