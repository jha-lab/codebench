# AccelBench

Cycle-accurate accelerator benchmarking framework for co-design.


## Usage
```shell
python run.py [model_name] [-c] [path to config file] [-e] [accelerator embedding]
```

For `accelerator embedding`, input a vector embedding for accelerator hyperparameters. It corresponds to `[Pib, Pix, Piy, Pof, Pkx, Pky, batch_size, activation_buffer_size, weight_buffer_size, mask_buffer_size, main_mem_type, main_mem_config]`.

As per the paper, here is the format for the main memory type and configuration:

>For on-chip buffer size, the unit is in MB.
For main memory type, choose among 1: RRAM, 2: DRAM, and 3: HBM.
For main memory configuration, choose among:
>- RRAM: 1: [16, 2, 2], 2: [8, 2, 4], 3: [4, 2, 8],  4: [2, 2, 16], 5: [32, 2, 1], 6: [1, 2, 32]
>- DRAM: 1: [16, 2, 2], 2: [8, 2, 4], 3: [32, 2, 1], 4: [16, 4, 1]
>- HMB: 1: [32, 1, 4].
The numbers corresponds to [#banks, #ranks, #channels].

The main `codebench/run_boshcode.py` script uses the `Accelerator` class for simulation of each accelerator within the AccelBench design space.
