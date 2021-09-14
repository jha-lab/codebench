import Accelerator
import argparse
import os
import time
import math

parser = argparse.ArgumentParser()
parser.add_argument('file', type = str, help = 'The file name of the frozen graph.')
parser.add_argument('--clk', type = int, default = 700, dest = "clk", help = 'The clock rate of the accelerator.')
parser.add_argument('--PE', type = int, default = 64, dest = "PE", help = 'The number of PEs. Needs to be a multiple of 32.')
parser.add_argument('--Lane', type = int, default = 72, dest = "Lane", help = 'The number of MacLanes per PE. Needs to be a multiple of 9.')
parser.add_argument('--batch', type = int, default = 1, dest = "batch", help = 'The number of batch size.')
parser.add_argument('--act', type = int, default = 12, dest = "act", help = 'The size of the activation buffer in MB.')
parser.add_argument('--wgt', type = int, default = 24, dest = "wgt", help = 'The size of the weight buffer in MB.')
parser.add_argument('--mask', type = int, default = 4, dest = "mask", help = 'The size of the mask buffer in MB.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

if args.clk:
    clk = args.clk

if args.PE:
    '''
    if args.PE % 16 != 0:
        parser.exit(1, 'The number of PEs needs to be a multiple of 16.\n')
    '''   
    PE = args.PE

if args.Lane:
    '''
    if args.Lane % 9 != 0:
        parser.exit(1, 'The number of Lanes needs to be a multiple of 9.\n')
    '''
    Lane = args.Lane

if args.batch:
    batch_size = args.batch

if args.act:
    Activation_buffer = args.act
if args.wgt:
    Weight_buffer = args.wgt
if args.mask:
    Mask_buffer = args.mask

    
### define parameters (original defines.py)
def defines(clk, PE, Lane, Activation_buffer, Weight_buffer, Mask_buffer):
    '''
    define all used parameters for architecture spec
    input: clk freq (MHz), #PE, #Mac_Lane per PE, Activation Buffer size, Weight Buffer size, Mask Buffer size (Buffer size unit: MB)
    output: all parameters
    '''
    # PE = Pib * Pix * Piy
    # Lane = Pof * Pkx * Pky
    Mac = 16
    IL = 4
    FL = 16
    activation_sparsity = 0.5
    weight_sparsity = 0.5
    overlap_factor = 0.8

    Pib = 2
    Pix = 8
    Piy = 4
    Pif = 16
    Pof = 8
    Pkx = 3
    Pky = 3
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
    
    # Module RTL parameters
    # area unit: um^2
    # power unit: mW
    MacLane_RTL_area = 8716.107541
    MacLane_RTL_dynamic = 4.9185 / 700 * clk
    MacLane_RTL_leakage = 0.2940754
    
    DataFlow_RTL_area =  75946.982008
    DataFlow_RTL_dynamic = 11.3711 / 700 * clk
    DataFlow_RTL_leakage = 2.7312
    
    DMA_RTL_area = 144.636242
    DMA_RTL_dynamic = 0.0335287 / 700 * clk
    DMA_RTL_leakage = 0.0063819
    
    FIFO_RTL_area = 959.892356
    FIFO_RTL_dynamic = 0.0794728 / 700 * clk
    FIFO_RTL_leakage = 0.0277679
    
    BatchNorm_RTL_area = 14654.491357
    BatchNorm_RTL_dynamic = 8.3914 / 700 * clk
    BatchNorm_RTL_leakage = 0.5323004
    
    Im2Col_RTL_area = 14307.182942
    Im2Col_RTL_dynamic = 423.0658
    Im2Col_RTL_leakage = 0.3963450
    
    Loss_RTL_area = 4223.411180
    Loss_RTL_dynamic = 0.9453281 / 700 * clk
    Loss_RTL_leakage = 0.1236274
    
    Pooling_RTL_area = 344.542129
    Pooling_RTL_dynamic = 0.0581307 / 700 * clk
    Pooling_RTL_leakage = 0.0105603
    
    PreSparsity_RTL_area = 6216.896914
    PreSparsity_RTL_dynamic = 0.2852968 / 700 * clk
    PreSparsity_RTL_leakage = 0.1507329
    
    PostSparsity_RTL_area = 1181.201447
    PostSparsity_RTL_dynamic = 0.2613971 / 700 * clk
    PostSparsity_RTL_leakage = 0.0324301
    
    Scalar_RTL_area = 19694.689797
    Scalar_RTL_dynamic = 4.3910 / 700 * clk
    Scalar_RTL_leakage = 0.7260926
    
    Transposer_RTL_area = 784.180314
    Transposer_RTL_dynamic = 0.1488154 / 700 * clk
    Transposer_RTL_leakage = 0.0180851
    
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
    
    Loss_area = Loss_RTL_area * PE
    Loss_dynamic = Loss_RTL_dynamic * PE
    Loss_leakage = Loss_RTL_leakage * PE
    Loss_width = Mac * PE       # width should scale with #PEs to reflect the increase in parallelism
    
    Pooling_area = Pooling_RTL_area * PE
    Pooling_dynamic = Pooling_RTL_dynamic * PE
    Pooling_leakage = Pooling_RTL_leakage * PE
    
    PreSparsity_area = PreSparsity_RTL_area * PE * Lane
    PreSparsity_dynamic = PreSparsity_RTL_dynamic * PE * Lane
    PreSparsity_leakage = PreSparsity_RTL_leakage * PE * Lane
    PreSparsity_width = Mac * 2
    
    PostSparsity_area = PostSparsity_RTL_area * PE * Lane
    PostSparsity_dynamic = PostSparsity_RTL_dynamic * PE * Lane
    PostSparsity_leakage = PostSparsity_RTL_leakage * PE * Lane
    PostSparsity_width = Mac * 2
    
    Scalar_area = Scalar_RTL_area * PE
    Scalar_dynamic = Scalar_RTL_dynamic * PE
    Scalar_leakage = Scalar_RTL_leakage * PE
    Scalar_width = Mac * PE     # width should scale with #PEs to reflect the increase in parallelism
    
    Transposer_area = Transposer_RTL_area * PE * Lane
    Transposer_dynamic = Transposer_RTL_dynamic * PE * Lane
    Transposer_leakage = Transposer_RTL_leakage * PE * Lane
    
    # Buffer parameters
    # access energy unit: nJ
    # leakage power unit: mW
    # area unit: mm^2
    Weight_energy = (0.42123 + 0.418051) / 2 * math.sqrt(Weight_buffer / 24)        # (dynamic read energy per access + dynamic write energy per access) / 2
    Weight_leakage = 315.607 * 16 / 24 * Weight_buffer                              # total leakage power per bank * num of banks
    Weight_area = 7.05842 / 24 * Weight_buffer
    
    Activation_energy = (0.275546 + 0.273688) / 2 * math.sqrt(Activation_buffer / 12)
    Activation_leakage = 185.939 * 16 / 12 * Activation_buffer
    Activation_area = 3.63967 / 12 * Activation_buffer
    
    Mask_energy = (0.164271 + 0.163756) / 2 * math.sqrt(Mask_buffer / 4)
    Mask_leakage = 119.098 * 8 / 4 * Mask_buffer
    Mask_area = 1.21026 / 4 * Mask_buffer

    # Main memory parameters
    # access energy unit: nJ
    # leakage power unit: mW
    MainMem_energy = 0
    MainMem_leakage = 0
    
    return(Pmac, Tile, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic, DataFlow_leakage, DataFlow_area,
           DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth,
           BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area,
           Loss_dynamic, Loss_leakage, Loss_area, Loss_width, Pooling_dynamic, Pooling_leakage, Pooling_area,
           PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width,
           PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width,
           Scalar_dynamic, Scalar_leakage, Scalar_area, Scalar_width, Transposer_dynamic, Transposer_leakage, Transposer_area,
           IL, FL, Activation_buffer, Activation_area, Activation_energy, Activation_leakage,
           Weight_buffer, Weight_area, Weight_energy, Weight_leakage, Mask_buffer, Mask_area, Mask_energy, Mask_leakage,
           activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size, matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage)



start_time = time.time()

Pmac, Tile, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic, DataFlow_leakage, DataFlow_area,\
DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth,\
BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area,\
Loss_dynamic, Loss_leakage, Loss_area, Loss_width, Pooling_dynamic, Pooling_leakage, Pooling_area,\
PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width,\
PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width,\
Scalar_dynamic, Scalar_leakage, Scalar_area, Scalar_width, Transposer_dynamic, Transposer_leakage, Transposer_area,\
IL, FL, Activation_buffer, Activation_area, Activation_energy, Activation_leakage,\
Weight_buffer, Weight_area, Weight_energy, Weight_leakage, Mask_buffer, Mask_area, Mask_energy, Mask_leakage,\
activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size, matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage = defines(clk, PE, Lane, Activation_buffer, Weight_buffer, Mask_buffer)

chip = Accelerator.Accelerator(Pmac, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic, DataFlow_leakage, DataFlow_area,
                               DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth,
                               BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area,
                               Loss_dynamic, Loss_leakage, Loss_area, Loss_width, Pooling_dynamic, Pooling_leakage, Pooling_area,
                               PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width,
                               PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width,
                               Scalar_dynamic, Scalar_leakage, Scalar_area, Scalar_width, Transposer_dynamic, Transposer_leakage, Transposer_area,
                               IL, FL, Activation_buffer, Activation_area, Activation_energy, Activation_leakage,
                               Weight_buffer, Weight_area, Weight_energy, Weight_leakage, Mask_buffer, Mask_area, Mask_energy, Mask_leakage,
                               activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size, matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage)
chip.FillQueue(args.file, Tile, batch_size)
Scalar_cycle, BatchNorm_cycle, Loss_cycle = chip.PreProcess(clk)
Memory_cycle, ConvMat_cycle = chip.Process(clk)
total_cycles, latency, area, dynamic_energy, leakage_energy = chip.Print(clk)
#chip.Print()

print('Scalar cycle:\t', Scalar_cycle)
print('BatchNorm cycle:\t', BatchNorm_cycle)
print('Loss cycle:\t', Loss_cycle)
print('Memory cycle:\t', Memory_cycle)
print('ConvMat cycle:\t', ConvMat_cycle)

print('Total cycles:\t', total_cycles)
print('Total execution time:\t', latency)
print('Total area:\t', area)
print('Dynamic energy consumption:\t', dynamic_energy)
print('Leakage energy sonsumption: ', leakage_energy)

print('Simulation time:', time.time() - start_time)
