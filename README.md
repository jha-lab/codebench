# CODEBench: A Neural Architecture and Hardware Accelerator Co-Design Framework

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.11.0-e74a2b)

This repository contains the tool **CODEBench** which can be used to generate and evaluate different CNN-accelerator pairs. It runs the **BOSHCODE** algorithm to obtain the best-performing pair for the given constraints and the selected design space.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository](#clone-this-repository)
  - [Setup python environment](#setup-python-environment)
- [Run hardware-software co-design](#run-hardware-software-co-design)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)

## Environment setup

### Clone this repository
```
git clone https://github.com/jha-lab/codebench.git
cd codebench
```
### Setup python environment  
* **PIP**
```
virtualenv cnnbench
source cnnbench/bin/activate
pip install -r cnnbench/requirements.txt
```  
* **CONDA**
```
conda env create -f cnnbench/environment.yaml
```

## Run hardware-software co-design
```
cd boshcode
python run_boshcode.py
```
All training scripts use bash and have been implemented using [SLURM](https://slurm.schedmd.com/documentation.html). This will have to be setup before running the experiments.

Other flags can be used to control the training procedure (check using `python run_boshcode.py --help`). This script uses the SLURM scheduler over mutiple compute nodes in a cluster (each cluster assumed to have 1 GPU, this can be changed in the script `job_scripts/job_worker.sh`). SLURM can also be used in scenarios where distributed nodes are not available.

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{tuli2022codebench,
      title={{CODEBench}: A Neural Architecture and Hardware Accelerator Co-Design Framework}, 
      author={Tuli, Shikhar and Li, Chia-Hao and Sharma, Ritvik and Jha, Niraj K.},
      year={2022},
      eprint={----.-----},
      archivePrefix={arXiv},
      primaryClass={--.--}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.