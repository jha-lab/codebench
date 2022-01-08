import Accelerator
import torch2blocks
import argparse
import os
import time
import math

from six.moves import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--graphlib_file',
    type=str,
    default='../cnn_design-space/cnnbench/dataset/dataset_test.json',
    help='path to load the CNN graphlib dataset')
parser.add_argument('--cnn_model_hash',
    type=str,
    help='hash of the CNN model to be simulated')
parser.add_argument('--config_file', 
    type = str, 
    default = '..cnn_design-space/cnnbench/configs/CIFAR10/config.yaml', 
    help = 'path to the config file')
parser.add_argument('--model_file', 
    type = str,  
    help = 'path to the save model results')
parser.add_argument('--embedding', 
    type = float, 
    nargs = '+', 
    default = [2, 16, 8, 4, 8, 3, 3, 1, 12, 24, 4, 1, 1], 
    help = "The vector embedding for accelerator hyperparameters.\
            The vector is corresponding to [Pib, Pif, Pix, Piy, Pof, Pkx, Pky, batch_size, activation_buffer, weight_buffer, mask_buffer, main_mem_type, main_mem_config].\
            For on-chip buffer size, the unit is in MB.\
            For main memory type, choose among 1: RRAM, 2: DRAM, and 3: HBM.\
            For main memory configuration, choose among \
            RRAM: 1: [16,2,2], 2: [8,2,4], 3: [4,2,8],  4: [2,2,16], 5: [32,2,1], 6: [1,2,32],\
            DRAM: 1: [16,2,2], 2: [8,2,4], 3: [32,2,1], 4: [16,4,1],\
            HMB: 1: [32,1,4].\
            The numbers corresponds to [#banks, #ranks, #channels].")
'''

parser.add_argument('model_name', 
    type = str, 
    help = 'The name of the model.\
            Should be one among [lenet, alexnet, vgg11, vgg13,\
            vgg16, vgg19, resnet18, resnet34,\
            resnet50, resnet101, resnet152,\
            shufflenet, mobilenet, googlenet,\
            inception, xception, efficientnet-b0,\
            efficientnet-b1, efficientnet-b2,\
            efficientnet-b3, efficientnet-b4,\
            efficientnet-b5, efficientnet-b6,\
            efficientnet-b7, efficientnet-l2]')

parser.add_argument('--clk', type = int, default = 700, dest = "clk", help = 'The clock rate of the accelerator.')


parser.add_argument('--PE', type = int, default = 64, help = 'The number of PEs. Needs to be a multiple of 32.')
parser.add_argument('--Lane', type = int, default = 1, help = 'The number of MacLanes per PE. Needs to be a multiple of 9.')
parser.add_argument('--batch', type = int, default = 1, help = 'The number of batch size.')
parser.add_argument('--act', type = int, default = 12, help = 'The size of the activation buffer in MB.')
parser.add_argument('--wgt', type = int, default = 24, help = 'The size of the weight buffer in MB.')
parser.add_argument('--mask', type = int, default = 4, help = 'The size of the mask buffer in MB.')
parser.add_argument('--mem_type', type = int, default = 1, help = 'The main memory type. 1: RRAM, 2: DRAM, 3: HBM')
parser.add_argument('--mem_config', type = int, default = 1, help = 'The choice of main memory configuration.\
                       RRAM: 1: [16,2,2], 2: [8,2,4], 3: [4,2,8],  4: [2,2,16], 5: [32,2,1], 6: [1,2,32]\
                       DRAM: 1: [16,2,2], 2: [8,2,4], 3: [32,2,1], 4: [16,4,1]\
                        HMB: 1: [32,1,4]\
                        [#banks, #ranks, #channels]')
'''
args = parser.parse_args()

assert len(args.embedding) == 13, "Some accelerator hyperparameters are missing."
assert args.embedding[10] in [1, 2, 3], "Memory type should be one of 1: RRAM, 2: DRAM, or 3: HBM."

if args.embedding[10] == 1:
    assert args.embedding[11] in [1, 2, 3, 4, 5, 6], "Memory configuration should be among RRAM: 1: [16,2,2], 2: [8,2,4], 3: [4,2,8],  4: [2,2,16], 5: [32,2,1], 6: [1,2,32]"
elif args.embedding[10] == 2:
    assert args.embedding[11] in [1, 2, 3, 4], "Memory configuration should be among DRAM: 1: [16,2,2], 2: [8,2,4], 3: [32,2,1], 4: [16,4,1]"
else:
    assert args.embedding[11] == 1, "Memory configuration should be HMB: 1: [32,1,4]"

clk = 700   # clock frequency (MHz)

Pib        = int(args.embedding[0])
Pif        = int(args.embedding[1])
Pix        = int(args.embedding[2])
Piy        = int(args.embedding[3])
Pof        = int(args.embedding[4])
Pkx        = int(args.embedding[5])
Pky        = int(args.embedding[6])
batch      = int(args.embedding[7])
act        = int(args.embedding[8])  # activation buffer size (MB)
wgt        = int(args.embedding[9])  # weight buffer size (MB)
mask       = args.embedding[10]       # mask buffer size (MB)
mem_type   = int(args.embedding[11])
mem_config = int(args.embedding[12])

### define parameters (original defines.py)
#def defines(clk, PE, Lane, Activation_buffer, Weight_buffer, Mask_buffer, Mem_type, Mem_config):
def defines():
    '''
    define all parameters used for architecture spec

    inputs:  clk freq (MHz), #PE, #Mac_Lane per PE, Activation Buffer size, Weight Buffer size, Mask Buffer size (Buffer size unit: MB)
    outputs: all parameters
    '''
    PE = Pib * Pix * Piy
    Lane = Pof * Pkx * Pky
    Mac = 16
    IL = 4
    FL = 16
    activation_sparsity = 0.5
    weight_sparsity = 0.5
    overlap_factor = 0.8

    #Pib = 2
    #Pix = 8
    #Piy = 4
    #Pif = 16
    #Pof = 1
    #Pkx = 1
    #Pky = 1
    Pmac = [Pib, Pix, Piy, Pif, Pof, Pkx, Pky]
    
    Tib = 4
    Tix = 64
    Tiy = 64
    Tif = 64
    Tile = [Tib, Tix, Tiy, Tif]
    
    # Memory parameters
    # unit: element
    data_size = 64
    cube_size = round(data_size ** (1.0 / 3))   # = 4
    matrix_size = int(math.sqrt(data_size))     # = 8
    # DDR_bandwidth unit: MB/s
    DDR_bandwidth = 40000
    # DRAM_bandwidth unit: bit/cycle
    DRAM_bandwidth = DDR_bandwidth * 1024 * 1024 * 8 * 1000 / clk * 1e-9

    # get the model ops to define the hardware
    ops, conv_shapes, head_shapes = torch2blocks.CNNBenchModel2Ops(args.config_file, args.graphlib_file, args.cnn_model_hash, batch)

    ReLU = False
    SiLU = False
    Upsampling = False

    for i in ops:
        if i[0] == 'ReLU' and not ReLU:
            ReLU = True
        elif i[0] == 'SiLU' and not SiLU:
            SiLU = True
        elif i[0] == 'UpsamplingBilinear2d' and not Upsampling:
            Upsampling = True
        else:
            continue
        if ReLU and SiLU and Upsampling:
            break

    # Module RTL parameters
    # area unit: um^2
    # power unit: mW
    if Pif == 16:
        if ReLU and not SiLU:
            MacLane_RTL_area = 8803.877125
            MacLane_RTL_dynamic = 4.7566 / 700 * clk
            MacLane_RTL_leakage = 0.2974811
        elif SiLU and not ReLU:
            MacLane_RTL_area = 9391.040997
            MacLane_RTL_dynamic = 5.0210 / 700 * clk
            MacLane_RTL_leakage = 0.3157760
        elif ReLU and SiLU:
            MacLane_RTL_area = 9435.567886
            MacLane_RTL_dynamic = 5.3176 / 700 * clk
            MacLane_RTL_leakage = 0.3241016

    elif Pif == 1:
        if ReLU and not SiLU:
            MacLane_RTL_area = 597.964253
            MacLane_RTL_dynamic = 0.2756146 / 700 * clk
            MacLane_RTL_leakage = 0.0223711
        elif SiLU and not ReLU:
            MacLane_RTL_area = 1209.955838
            MacLane_RTL_dynamic = 0.6705926 / 700 * clk
            MacLane_RTL_leakage = 0.0419121
        elif ReLU and SiLU:
            MacLane_RTL_area = 1199.509430
            MacLane_RTL_dynamic = 0.6896749 / 700 * clk
            MacLane_RTL_leakage = 0.0415079

    DataFlow_RTL_area =  75946.982008
    DataFlow_RTL_dynamic = 11.3711 / 700 * clk
    DataFlow_RTL_leakage = 2.7312
    
    DMA_RTL_area = 144.636242
    DMA_RTL_dynamic = 0.0335287 / 700 * clk
    DMA_RTL_leakage = 0.0063819
    
    FIFO_RTL_area = 484.379110
    FIFO_RTL_dynamic = 0.0563909 / 700 * clk
    FIFO_RTL_leakage = 0.0140774
    
    BatchNorm_RTL_area = 14654.491357
    BatchNorm_RTL_dynamic = 8.3914 / 700 * clk
    BatchNorm_RTL_leakage = 0.5323004
    
    Im2Col_RTL_area = 14307.182942
    Im2Col_RTL_dynamic = 423.0658
    Im2Col_RTL_leakage = 0.3963450
    
    '''
    Loss_RTL_area = 4223.411180
    Loss_RTL_dynamic = 0.9453281 / 700 * clk
    Loss_RTL_leakage = 0.1236274
    '''
    
    Pooling_RTL_area = 3395.049675
    Pooling_RTL_dynamic = 1.1430 / 700 * clk
    Pooling_RTL_leakage = 0.1281938
    
    PreSparsity_RTL_area = 6216.896914
    PreSparsity_RTL_dynamic = 0.2852968 / 700 * clk
    PreSparsity_RTL_leakage = 0.1507329
    
    PostSparsity_RTL_area = 1181.201447
    PostSparsity_RTL_dynamic = 0.2613971 / 700 * clk
    PostSparsity_RTL_leakage = 0.0324301

    GlobalAvgPooling_RTL_area = 0     # GlobalAvgPooling module is used in pooling module to perform avg pooling, so the area is counted in pooling_RTL_area already
    GlobalAvgPooling_RTL_dynamic = 1.2148 / 700 * clk
    GlobalAvgPooling_RTL_leakage = 0.0848479

    Upsampling_RTL_area = 749.704715
    Upsampling_RTL_dynamic = 0.1503713 / 700 * clk
    Upsampling_RTL_leakage = 0.0293699
    
    # Module scaled parameters
    MacLane_area = MacLane_RTL_area * PE * Lane
    MacLane_dynamic = MacLane_RTL_dynamic * PE * Lane
    MacLane_leakage = MacLane_RTL_leakage * PE * Lane
    
    DataFlow_area =  DataFlow_RTL_area * PE
    DataFlow_dynamic = DataFlow_RTL_dynamic * PE
    DataFlow_leakage = DataFlow_RTL_leakage * PE
    
    DMA_area = DMA_RTL_area * PE
    DMA_dynamic = DMA_RTL_dynamic * PE
    DMA_leakage = DMA_RTL_leakage * PE
    DMA_bandwidth = int(data_size * 8 * PE / (IL + FL))
    
    FIFO_area = FIFO_RTL_area * PE * Lane * Mac * 2         # times the number of input data (activations + weights = Mac * 2)
    FIFO_dynamic = FIFO_RTL_dynamic * PE * Lane * Mac * 2   # times the number of input data (activations + weights = Mac * 2)
    FIFO_leakage = FIFO_RTL_leakage * PE * Lane * Mac * 2   # times the number of input data (activations + weights = Mac * 2)
    FIFO_depth = Mac * 2
    
    BatchNorm_area = BatchNorm_RTL_area * PE
    BatchNorm_dynamic = BatchNorm_RTL_dynamic * PE
    BatchNorm_leakage = BatchNorm_RTL_leakage * PE
    BatchNorm_width = Mac * PE  # width should scale with #PEs to reflect the increase in parallelism
    
    Im2Col_area = Im2Col_RTL_area * PE * Lane
    Im2Col_dynamic = Im2Col_RTL_dynamic * PE * Lane
    Im2Col_leakage = Im2Col_RTL_leakage * PE * Lane
    
    '''
    Loss_area = Loss_RTL_area * PE
    Loss_dynamic = Loss_RTL_dynamic * PE
    Loss_leakage = Loss_RTL_leakage * PE
    Loss_width = Mac * PE       # width should scale with #PEs to reflect the increase in parallelism
    '''
    
    Pooling_area = Pooling_RTL_area * PE
    Pooling_dynamic = Pooling_RTL_dynamic * PE
    Pooling_leakage = Pooling_RTL_leakage * PE
    Pooling_depth = 25 * PE
    
    PreSparsity_area = PreSparsity_RTL_area * PE * Lane
    PreSparsity_dynamic = PreSparsity_RTL_dynamic * PE * Lane
    PreSparsity_leakage = PreSparsity_RTL_leakage * PE * Lane
    PreSparsity_width = Mac * 2
    
    PostSparsity_area = PostSparsity_RTL_area * PE * Lane
    PostSparsity_dynamic = PostSparsity_RTL_dynamic * PE * Lane
    PostSparsity_leakage = PostSparsity_RTL_leakage * PE * Lane
    PostSparsity_width = Mac * 2
    
    GlobalAvgPooling_area = GlobalAvgPooling_RTL_area * PE
    GlobalAvgPooling_dynamic = GlobalAvgPooling_RTL_dynamic * PE
    GlobalAvgPooling_leakage = GlobalAvgPooling_RTL_leakage * PE
    GlobalAvgPooling_depth = 25 * PE
    
    Upsampling_area = Upsampling_RTL_area * PE
    Upsampling_dynamic = Upsampling_RTL_dynamic * PE
    Upsampling_leakage = Upsampling_RTL_leakage * PE
    Upsampling_depth = 4 * PE

    # Buffer parameters
    # access energy unit: nJ
    # leakage power unit: mW
    # area unit: mm^2
    Weight_energy = (0.42123 + 0.418051) / 2 * math.sqrt(wgt / 24)    # (dynamic read energy per access + dynamic write energy per access) / 2
    Weight_leakage = 315.607 * 16 / 24 * wgt                                    # total leakage power per bank * num of banks
    Weight_area = 7.05842 / 24 * wgt
    
    Activation_energy = (0.275546 + 0.273688) / 2 * math.sqrt(act / 12)
    Activation_leakage = 185.939 * 16 / 12 * act
    Activation_area = 3.63967 / 12 * act
    
    Mask_energy = (0.164271 + 0.163756) / 2 * math.sqrt(mask / 4)
    Mask_leakage = 119.098 * 8 / 4 * mask
    Mask_area = 1.21026 / 4 * mask

    # Main memory parameters
    # access energy unit: nJ
    # leakage power unit: mW
    if mem_type == 1:
        if mem_config == 1:
            MainMem_energy = 17.321165
        elif mem_config == 2:
            MainMem_energy = 32.655499
        elif mem_config == 3:
            MainMem_energy = 69.3328008
        elif mem_config == 4:
            MainMem_energy = 149.0564671
        elif mem_config == 5:
            MainMem_energy = 8.63281599
        else:
            MainMem_energy = 340.9690911
    elif mem_type == 2:
        if mem_config == 1:
            MainMem_energy = 69.00706337
        elif mem_config == 2:
            MainMem_energy = 200.2819855
        elif mem_config == 3:
            MainMem_energy = 53.43777522
        else:
            MainMem_energy = 38.28777884
    else:
        MainMem_energy = 26.5569841   
            
    MainMem_leakage = 0
    
    return(Pmac, Tile, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic, DataFlow_leakage, DataFlow_area,
           DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth,
           BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area,
           Pooling_dynamic, Pooling_leakage, Pooling_area, Pooling_depth,
           GlobalAvgPooling_dynamic, GlobalAvgPooling_leakage, GlobalAvgPooling_area, GlobalAvgPooling_depth,
           Upsampling_dynamic, Upsampling_leakage, Upsampling_area, Upsampling_depth,
           PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width,
           PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width,
           IL, FL, act, Activation_area, Activation_energy, Activation_leakage,
           wgt, Weight_area, Weight_energy, Weight_leakage, mask, Mask_area, Mask_energy, Mask_leakage,
           activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size, matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage,
           ops, conv_shapes, head_shapes)



start_time = time.time()

Pmac, Tile, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic, DataFlow_leakage, DataFlow_area,\
DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth,\
BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area,\
Pooling_dynamic, Pooling_leakage, Pooling_area, Pooling_depth,\
GlobalAvgPooling_dynamic, GlobalAvgPooling_leakage, GlobalAvgPooling_area, GlobalAvgPooling_depth,\
Upsampling_dynamic, Upsampling_leakage, Upsampling_area, Upsampling_depth,\
PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width,\
PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width,\
IL, FL, act, Activation_area, Activation_energy, Activation_leakage,\
wgt, Weight_area, Weight_energy, Weight_leakage, mask, Mask_area, Mask_energy, Mask_leakage,\
activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size, matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage,\
ops, conv_shapes, head_shapes = defines()

chip = Accelerator.Accelerator(Pmac, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic, DataFlow_leakage, DataFlow_area,
                               DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth,
                               BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area,
                               Pooling_dynamic, Pooling_leakage, Pooling_area, Pooling_depth,
                               GlobalAvgPooling_dynamic, GlobalAvgPooling_leakage, GlobalAvgPooling_area, GlobalAvgPooling_depth,
                               Upsampling_dynamic, Upsampling_leakage, Upsampling_area, Upsampling_depth,
                               PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width,
                               PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width,
                               IL, FL, act, Activation_area, Activation_energy, Activation_leakage,
                               wgt, Weight_area, Weight_energy, Weight_leakage, mask, Mask_area, Mask_energy, Mask_leakage,
                               activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size, matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage)
chip.FillQueue(Tile, batch, ops, conv_shapes, head_shapes)
BatchNorm_cycle, Pooling_cycle, GlobalAvgPooling_cycle, Upsampling_cycle = chip.PreProcess(clk)
Memory_cycle, ConvMat_cycle = chip.Process(clk)
total_cycles, latency, area, dynamic_energy, leakage_energy = chip.Print(clk)
#chip.Print()

print(f'BatchNorm cycle: \t {BatchNorm_cycle}')
#print(f'Loss cycle: \t\t {Loss_cycle}')
print(f'Pooling cycle: \t\t {Pooling_cycle}')
print(f'GlobalAvgPooling cycle:  {GlobalAvgPooling_cycle}')
print(f'Upsampling cycle: \t {Upsampling_cycle}')
print()
print(f'Memory cycle: \t{Memory_cycle}')
print(f'ConvMat cycle: \t{ConvMat_cycle}')
print()
print(f'Total cycles: \t\t\t{total_cycles}')
print(f'Total execution time: \t\t{latency}')
print(f'Total area: \t\t\t{area}')
print(f'Dynamic energy consumption: \t{dynamic_energy}')
print(f'Leakage energy sonsumption: \t{leakage_energy}')
print()
print('Simulation time:', time.time() - start_time)

# Saving results to a pickle file
pickle.dump({'bacthnorm_cycles': BatchNorm_cycle,
    #'loss_cycles': Loss_cycle,
    'pool_cycles': Pooling_cycle,
    'globalavgpool_cycles': GlobalAvgPooling_cycle,
    'upsampling_cycles': Upsampling_cycle,
    'memory_cycles': Memory_cycle,
    'conv_cycles': ConvMat_cycle,
    'total_cycles': total_cycles, 
    'latency': latency, 
    'area': area, 
    'dynamic_energy': dynamic_energy, 
    'leakage_energy': leakage_energy}, open(args.model_file, 'wb+'), pickle.HIGHEST_PROTOCOL)
