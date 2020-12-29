#!/bin/bash

numNodes=1
numTasks=4
cluster="adroit"
gpu="tesla_v100"

# while test $# -gt 0; do
# case "$1" in
#     -numNodes)
#         shift
#         $numNodes=$1
#         shift
#         ;;
#     -numTasks)
#         shift
#         $numTasks=$1
#         shift
#         ;;
#     -cluster)
#         shift
#         $cluster=$1
#         shift
#         ;;
#     *)
#        echo "Unrecognized flag -$1"
#        return 1;
#        ;;
# esac
# done  

# if [[ $cluster -eq "adroit" ]]
# then
# 	$gpu="gpu:tesla_v100:1"
# elif [[ $cluster -eq "tiger" ]]
# then
# 	$gpu="gpu:1"
# else
# 	echo "Unrecognized cluster"
# 	return 1
# fi

#SBATCH --job-name=nasbench_multi           # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=4                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:tesla_v100:1                 # number of gpus per node
#SBATCH --time=00:50:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate acc-nn

numVertices=2

python generate_graphs_script.py --max_vertices $numVertices

for i in {0..($numTasks-1)}
do
	srun -N 1 -n 1 python run_evaluation_script.py --worker_id i --total_workers $numTasks --module_vertices $numVertices &
done

wait

python cleanup_script.py