#!/bin/bash

# Export all variables to environment
set -a

numNodes=1
numTasks=2
cluster="adroit"
cluster_gpu="gpu:tesla_v100:2"
numVertices=2

Help()
{
   # Display Help
   echo "Flags for this script"
   echo
   echo "Syntax: scriptTemplate [-numNodes|numTasks|cluster|vertices]"
   echo "options:"
   echo "-numNodes [default = 1]				    Number of nodes to use in cluster"
   echo "-numTasks [default = 2]				    Number of tasks across all nodes"
   echo "-cluster [default = \"adroit\"]		Selected cluster - adroit or tiger"
   echo "-vertices [default = 2]				    Number of vertices per module in NASBench"
   echo "-help									            Call this help message"
   echo
}

while test $# -gt 0; do
case "$1" in
    -numNodes)
        shift
        $numNodes=$1
        shift
        ;;
    -numTasks)
        shift
        $numTasks=$1
        shift
        ;;
    -cluster)
        shift
        $cluster=$1
        shift
        ;;
    -vertices)
        shift
        $cluster=$1
        shift
        ;;    
    -help)
       Help
       return 1;
       ;;
    *)
       echo "Unrecognized flag -$1"
       return 1;
       ;;
esac
done  

if [[ $cluster -eq "adroit" ]]
then
  if [[ $numNodes -gt 1 ]]
	then
    cluster_gpu="gpu:tesla_v100:4"
  else
    cluster_gpu="gpu:tesla_v100:"+$numTasks
  fi
elif [[ $cluster -eq "tiger" ]]
then
  if [[ $numNodes -gt 1 ]]
  then
    cluster_gpu="gpu:4"
  else
    cluster_gpu="gpu:"+$numTasks
  fi
else
	echo "Unrecognized cluster"
	return 1
fi

numTask_end=$(($numTasks-1))

job_file="job_nasbench_n${numNodes}_t${numTasks}_v${numVertices}.slurm"

echo "#!/bin/bash
#SBATCH --job-name=nasbench_multi           # create a short name for your job
#SBATCH --nodes=${numNodes}                 # node count
#SBATCH --ntasks=${numTasks}                # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=${cluster_gpu}               # number of gpus per node
#SBATCH --time=10:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate acc-nn

python generate_graphs_script.py --max_vertices $numVertices

for i in {0..${numTask_end}}
do
  python run_evaluation_script.py --worker_id $i --total_workers ${numTasks} --module_vertices ${numVertices} &
done

wait

python cleanup_script.py" > $job_file

sbatch $job_file