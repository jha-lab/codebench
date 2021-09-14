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
MacLane_RTL_area = 8716.107541

DataFlow_RTL_area =  75946.982008

DMA_RTL_area = 144.636242

FIFO_RTL_area = 959.892356

BatchNorm_RTL_area = 14654.491357

Im2Col_RTL_area = 14307.182942

Loss_RTL_area = 4223.411180

Pooling_RTL_area = 344.542129

PreSparsity_RTL_area = 6216.896914

PostSparsity_RTL_area = 1181.201447

Scalar_RTL_area = 19694.689797

Transposer_RTL_area = 784.180314

# Module scaled parameters
MacLane_area = MacLane_RTL_area * PE * Lane

DataFlow_area =  DataFlow_RTL_area * PE

DMA_area = DMA_RTL_area * PE

FIFO_area = FIFO_RTL_area * PE * Lane * Mac * 2

BatchNorm_area = BatchNorm_RTL_area * PE

Im2Col_area = Im2Col_RTL_area * PE * Lane

Loss_area = Loss_RTL_area * PE

Pooling_area = Pooling_RTL_area * PE

PreSparsity_area = PreSparsity_RTL_area * PE * Lane

PostSparsity_area = PostSparsity_RTL_area * PE * Lane

Scalar_area = Scalar_RTL_area * PE

Transposer_area = Transposer_RTL_area * PE * Lane

# Buffer parameters
# access energy unit: nJ
# leakage power unit: mW
# area unit: mm^2
Weight_area = 7.05842 / 24 * Weight_buffer

Activation_area = 3.63967 / 12 * Activation_buffer

Mask_area = 1.21026 / 4 * Mask_buffer

PE_area = MacLane_area + FIFO_area + BatchNorm_area + Loss_area + Pooling_area + PreSparsity_area + PostSparsity_area + Scalar_area + Transposer_area + DataFlow_area + Im2Col_area

Buffer_area = Weight_area + Activation_area + Mask_area
Total_area = (PE_area + DMA_area) * 1e-6 + Buffer_area

print('PE:\t', PE_area*1e-6)
print('DMA:\t', DMA_area*1e-6)
print('PE+DMA:\t', (PE_area+DMA_area)*1e-6)
print('Buffer:\t', Buffer_area)
print('Total:\t', Total_area)
