#!/bin/bash

# Script to train a given model

# Author : Shikhar Tuli

cluster="tiger"
id="stuli"
autotune="0"
train_cnn="0"
cnn_model_hash=""
cnn_model_dir=""
cnn_config_file=""
neighbor_file=""
dataset=""
graphlib_file=""
accel_hash=""
accel_emb=""
accel_model_file=""

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Script to train a given model${ENDC}"
   echo
   echo -e "Syntax: source ${CYAN}./job_scripts/job_train.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"tiger\"${ENDC}] \t\t Selected cluster - adroit, tiger or della"
   echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"stuli\"${ENDC}] \t\t\t Selected PU-NetID to email slurm updates"
   echo -e "${YELLOW}-a${ENDC} | ${YELLOW}--autotune${ENDC} [default = ${GREEN}\"0\"${ENDC}] \t\t To autotune the given CNN model"
   echo -e "${YELLOW}-t${ENDC} | ${YELLOW}--train_cnn${ENDC} [default = ${GREEN}\"0\"${ENDC}] \t\t To train the given CNN model"
   echo -e "${YELLOW}-m${ENDC} | ${YELLOW}--cnn_model_hash${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t CNN Model hash"
   echo -e "${YELLOW}-d${ENDC} | ${YELLOW}--cnn_model_dir${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Directory to save the CNN model"
   echo -e "${YELLOW}-f${ENDC} | ${YELLOW}--cnn_config_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to the CNN config file"
   echo -e "${YELLOW}-n${ENDC} | ${YELLOW}--neighbor_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to the chosen neighbor"
   echo -e "${YELLOW}-s${ENDC} | ${YELLOW}--dataset${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t\t Dataset name"
   echo -e "${YELLOW}-g${ENDC} | ${YELLOW}--graphlib_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to the graphlib dataset"
   echo -e "${YELLOW}-a${ENDC} | ${YELLOW}--accel_hash${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Hash of the CNN-Accelerator pair"
   echo -e "${YELLOW}-e${ENDC} | ${YELLOW}--accel_emb${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Embedding of the Accelerator"
   echo -e "${YELLOW}-l${ENDC} | ${YELLOW}--accel_model_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Path to save Accelerator result"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -c | --cluster)
        shift
        cluster=$1
        shift
        ;;
    -i | --id)
        shift
        id=$1
        shift
        ;;
    -a | --autotune)
        shift
        autotune=$1
        shift
        ;;
    -t | --train_cnn)
        shift
        train_cnn=$1
        shift
        ;;
    -m | --cnn_model_hash)
        shift
        cnn_model_hash=$1
        shift
        ;;
    -d | --cnn_model_dir)
        shift
        cnn_model_dir=$1
        shift
        ;;
    -f | --cnn_config_file)
        shift
        cnn_config_file=$1
        shift
        ;;
    -n | --neighbor_file)
        shift
        neighbor_file=$1
        shift
        ;;
    -s | --dataset)
        shift
        dataset=$1
        shift
        ;;
    -g | --graphlib_file)
        shift
        graphlib_file=$1
        shift
        ;;
    -a | --accel_hash)
        shift
        accel_hash=$1
        shift
        ;;
    -e | --accel_emb)
        shift
        accel_emb=$1
        shift
        ;;
    -l | --accel_model_file)
        shift
        accel_model_file=$1
        shift
        ;;
    -h| --help)
       Help
       return 1;
       ;;
    *)
       echo "Unrecognized flag $1"
       return 1;
       ;;
esac
done  

if [[ $cluster == "adroit" ]]
then
  cluster_gpu="gpu:tesla_v100:4"
elif [[ $cluster == "tiger" ]]
then
  cluster_gpu="gpu:4"
elif [[ $cluster == "della" ]]
then
  cluster_gpu="gpu:2"
else
	echo "Unrecognized cluster"
	return 1
fi

job_file="./job_${accel_hash}.slurm"
mkdir -p "./job_scripts/${dataset}/"

cd "./job_scripts/${dataset}/"

# Create SLURM job script to train CNN-Accelerator pair
echo "#!/bin/bash" >> $job_file
echo "#SBATCH --job-name=code_${dataset}_${accel_hash}       # create a short name for your job" >> $job_file
echo "#SBATCH --nodes=1                                      # node count" >> $job_file
echo "#SBATCH --ntasks=1                                     # total number of tasks across all nodes" >> $job_file
echo "#SBATCH --cpus-per-task=24                             # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
# echo "#SBATCH --cpus-per-task=2                              # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
echo "#SBATCH --mem=128G                                     # memory per cpu-core (4G is default)" >> $job_file
if [[ $train_cnn == "1" ]]
then
    echo "#SBATCH --gres=${cluster_gpu}                      # number of gpus per node" >> $job_file
    # echo "#SBATCH --gres=gpu:1" >> $job_file
fi
echo "#SBATCH --time=144:00:00                               # total run time limit (HH:MM:SS)" >> $job_file
# echo "#SBATCH --time=4:00:00                               # total run time limit (HH:MM:SS)" >> $job_file
# echo "#SBATCH --mail-type=all                                # send email" >> $job_file
# echo "#SBATCH --mail-user=stuli@princeton.edu" >> $job_file
echo "" >> $job_file
echo "module purge" >> $job_file
echo "module load anaconda3/2020.7" >> $job_file
# echo "conda init bash" >> $job_file
echo "conda activate cnnbench" >> $job_file
echo "" >> $job_file
echo "cd ../.." >> $job_file
echo "" >> $job_file
echo "export MKL_SERVICE_FORCE_INTEL=1" >> $job_file
echo "python ../accelerator_design-space/accelbench/run.py --config_file ${cnn_config_file} \
    --graphlib_file ${graphlib_file} \
    --cnn_model_hash ${cnn_model_hash} \
    --embedding ${accel_emb} \
    --model_file ${accel_model_file} &" >> $job_file
# echo "python -c \"import random, os; \
#     from six.moves import cPickle as pickle; \
#     latency = random.random(); \
#     area = 10 * random.random(); \
#     dynamic_energy = random.random(); \
#     leakage_energy = random.random(); \
#     pickle.dump({'latency': latency, 'area': area, 'dynamic_energy': dynamic_energy, 'leakage_energy': leakage_energy}, \
#     open('${accel_model_file}', 'wb+'), pickle.HIGHEST_PROTOCOL)\" &" >> $job_file
if [[ $train_cnn == "1" ]]
then
    if [[ $neighbor_file == "" ]]
    then
        echo "python ../cnn_design-space/cnnbench/model_trainer.py --config_file ${cnn_config_file} \
          --graphlib_file ${graphlib_file} \
          --model_dir ${cnn_model_dir} \
          --model_hash ${cnn_model_hash} \
          --autotune ${autotune}" >> $job_file
    else
        echo "python ../cnn_design-space/cnnbench/model_trainer.py --config_file ${cnn_config_file} \
          --graphlib_file ${graphlib_file} \
          --neighbor_file ${neighbor_file} \
          --model_dir ${cnn_model_dir} \
          --model_hash ${cnn_model_hash} \
          --autotune ${autotune}" >> $job_file
    fi
fi
# echo "python -c \"import time, torch, random, os, numpy; \
#     acc = random.random(); \
#     ckpt = {'train_accuracies': [acc], 'val_accuracies': [acc], 'test_accuracies': [acc]}; \
#     os.makedirs('${cnn_model_dir}'); \
#     torch.save(ckpt, os.path.join('${cnn_model_dir}', 'model.pt'))\" &" >> $job_file

# Commenting out "wait" since the AccelBench simulation should finish way before CNNBench
# echo "wait" >> $job_file

sbatch $job_file