latency = 0.02812615428571429

clk = 700
PE = 64
Lane = 72
Mac = 16

Weight_buffer = 24
Activation_buffer = 12
Mask_buffer = 4   

# Module RTL parameters
# area unit: um^2
# power unit: mW
MacLane_RTL_leakage = 0.2940754
    
DataFlow_RTL_leakage = 2.7312

DMA_RTL_leakage = 0.0063819

FIFO_RTL_leakage = 0.0277679

BatchNorm_RTL_leakage = 0.6338611

Im2Col_RTL_leakage = 0.3963450

Loss_RTL_leakage = 0.1236274

Pooling_RTL_leakage = 0.0105603

PreSparsity_RTL_leakage = 0.1507329

PostSparsity_RTL_leakage = 0.0324301

Scalar_RTL_leakage = 0.7260926

Transposer_RTL_leakage = 0.0180851

# Module scaled parameters

MacLane_leakage = MacLane_RTL_leakage * PE * Lane

DataFlow_leakage = DataFlow_RTL_leakage * PE

DMA_leakage = DMA_RTL_leakage * PE

FIFO_leakage = FIFO_RTL_leakage * PE * Lane

BatchNorm_leakage = BatchNorm_RTL_leakage * PE

Im2Col_leakage = Im2Col_RTL_leakage * PE * Lane

Loss_leakage = Loss_RTL_leakage * PE

Pooling_leakage = Pooling_RTL_leakage * PE

PreSparsity_leakage = PreSparsity_RTL_leakage * PE * Lane

PostSparsity_leakage = PostSparsity_RTL_leakage * PE * Lane

Scalar_leakage = Scalar_RTL_leakage * PE

Transposer_leakage = Transposer_RTL_leakage * PE * Lane

# Buffer parameters
# access energy unit: nJ
# leakage power unit: mW
# area unit: mm^2

Weight_leakage = 315.607 * 16 / 24 * Weight_buffer                              # total leakage power per bank * num of banks

Activation_leakage = 185.939 * 16 / 12 * Activation_buffer

Mask_leakage = 119.098 * 8 / 4 * Mask_buffer


PE_leakage_power = MacLane_leakage + FIFO_leakage + BatchNorm_leakage + Loss_leakage + Pooling_leakage + PreSparsity_leakage + PostSparsity_leakage + Scalar_leakage + Transposer_leakage + DataFlow_leakage + Im2Col_leakage + DMA_leakage
Buffer_leakage_power = Weight_leakage + Activation_leakage + Mask_leakage

Total_leakage_power = (PE_leakage_power + Buffer_leakage_power) * latency

print('PE:\t', PE_leakage_power)
print('Buffer:\t', Buffer_leakage_power)

print('Total:\t', Total_leakage_power)
