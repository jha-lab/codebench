import math

# architecture spec
# buffer size unit: MB
# frequency unit: MHz
PE = 64
Lane = 72
Mac = 16
Weight_buffer = 24
Activation_buffer = 12
Mask_buffer = 4
clk = 700

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

batch_size = 32

# area unit: um^2
# power unit: mW

# Memory parameters
# unit: element
data_size = 64
cube_size = int(data_size ** (1.0 / 3))
matrix_size = int(math.sqrt(data_size))
# DDR_bandwidth unit: MB/s
DDR_bandwidth = 40000
# DRAM_bandwidth unit: bit/cycle
DRAM_bandwidth = DDR_bandwidth * 1024 * 1024 * 8 * 1000 / clk * 1e-9

# Module RTL parameters
MacLane_RTL_area = 8801.498079
MacLane_RTL_dynamic = 4.7925
MacLane_RTL_leakage = 0.297768

DataFlow_RTL_area =  75946.982008
DataFlow_RTL_dynamic = 11.3711
DataFlow_RTL_leakage = 2.7312

DMA_RTL_area = 144.636242
DMA_RTL_dynamic = 0.0335287
DMA_RTL_leakage = 0.0063819

FIFO_RTL_area = 603.693717
FIFO_RTL_dynamic = 0.0549122
FIFO_RTL_leakage = 0.0168573

BatchNorm_RTL_area = 17757.140433
BatchNorm_RTL_dynamic = 9.6513
BatchNorm_RTL_leakage = 0.6338611

Im2Col_RTL_area = 14307.182942
Im2Col_RTL_dynamic = 423.0658
Im2Col_RTL_leakage = 0.3963450

Loss_RTL_area = 4223.411180
Loss_RTL_dynamic = 0.9453281
Loss_RTL_leakage = 0.1236274

Pooling_RTL_area = 344.542129
Pooling_RTL_dynamic = 0.0581307
Pooling_RTL_leakage = 0.0105603

PreSparsity_RTL_area = 6216.896914
PreSparsity_RTL_dynamic = 0.2852968
PreSparsity_RTL_leakage = 0.1507329

PostSparsity_RTL_area = 1181.201447
PostSparsity_RTL_dynamic = 0.2613971
PostSparsity_RTL_leakage = 0.0324301

Scalar_RTL_area = 19694.689797
Scalar_RTL_dynamic = 4.3910
Scalar_RTL_leakage = 0.7260926

Transposer_RTL_area = 784.180314
Transposer_RTL_dynamic = 0.1488154
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
DMA_bandwidth = int(64 * 8 * PE / (IL + FL))

FIFO_area = FIFO_RTL_area * PE * Lane
FIFO_dynamic = FIFO_RTL_dynamic * PE * Lane
FIFO_leakage = FIFO_RTL_leakage * PE * Lane
FIFO_depth = 32

BatchNorm_area = BatchNorm_RTL_area * PE
BatchNorm_dynamic = BatchNorm_RTL_dynamic * PE
BatchNorm_leakage = BatchNorm_RTL_leakage * PE
BatchNorm_width = 16

Im2Col_area = Im2Col_RTL_area * PE * Lane
Im2Col_dynamic = Im2Col_RTL_dynamic * PE * Lane
Im2Col_leakage = Im2Col_RTL_leakage * PE * Lane

Loss_area = Loss_RTL_area * PE
Loss_dynamic = Loss_RTL_dynamic * PE
Loss_leakage = Loss_RTL_leakage * PE
Loss_width = 16

Pooling_area = Pooling_RTL_area * PE
Pooling_dynamic = Pooling_RTL_dynamic * PE
Pooling_leakage = Pooling_RTL_leakage * PE

PreSparsity_area = PreSparsity_RTL_area * PE * Lane
PreSparsity_dynamic = PreSparsity_RTL_dynamic * PE * Lane
PreSparsity_leakage = PreSparsity_RTL_leakage * PE * Lane
PreSparsity_width = 32

PostSparsity_area = PostSparsity_RTL_area * PE * Lane
PostSparsity_dynamic = PostSparsity_RTL_dynamic * PE * Lane
PostSparsity_leakage = PostSparsity_RTL_leakage * PE * Lane
PostSparsity_width = 32

Scalar_area = Scalar_RTL_area * PE
Scalar_dynamic = Scalar_RTL_dynamic * PE
Scalar_leakage = Scalar_RTL_leakage * PE
Scalar_width = 16

Transposer_area = Transposer_RTL_area * PE * Lane
Transposer_dynamic = Transposer_RTL_dynamic * PE * Lane
Transposer_leakage = Transposer_RTL_leakage * PE * Lane

# Buffer parameters
# access energy unit: nJ
# leakage power unit: mW
# area unit: mm^2
Weight_energy = (0.42123 + 0.418051) / 2
Weight_leakage = 315.607 * 16
Weight_area = 7.05842

Activation_energy = (0.275546 + 0.273688) / 2
Activation_leakage = 185.939 * 16
Activation_area = 3.63967

Mask_energy = (0.164271 + 0.163756) / 2
Mask_leakage = 119.098 * 8
Mask_area = 1.21026
