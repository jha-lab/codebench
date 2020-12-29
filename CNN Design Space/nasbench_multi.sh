#!/bin/bash

# Export all variables to environment
set -a

numNodes=1
numTasks=4
cluster="adroit"
cluster_gpu="gpu:tesla_v100:1"
numVertices=2

Help()
{
   # Display Help
   echo "Flags for this script"
   echo
   echo "Syntax: scriptTemplate [-numNodes|numTasks|cluster|vertices]"
   echo "options:"
   echo "-numNodes [default = 1]				Number of nodes to use in cluster"
   echo "-numTasks [default = 4]				Number of tasks across all nodes"
   echo "-cluster [default = \"adroit\"]		Selected cluster - adroit or tiger"
   echo "-vertices [default = 2]				Number of vertices per module in NASBench"
   echo "-help									Call this help message"
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
	cluster_gpu="gpu:tesla_v100:1"
elif [[ $cluster -eq "tiger" ]]
then
	cluster_gpu="gpu:1"
else
	echo "Unrecognized cluster"
	return 1
fi

sbatch job_nasbench.slurm