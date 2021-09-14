import pb2blocks
import Blocks
import IssueQueue
import Hardware
# import defines
import numpy as np

class Accelerator(object):
	def __init__(self,Pmac, MacLane_dynamic, MacLane_leakage, MacLane_area, DataFlow_dynamic,\
	DataFlow_leakage, DataFlow_area, DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth, FIFO_dynamic,\
	FIFO_leakage, FIFO_area, FIFO_depth, BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area,\
	BatchNorm_width, Im2Col_dynamic, Im2Col_leakage, Im2Col_area, Loss_dynamic, Loss_leakage, Loss_area,\
	Loss_width, Pooling_dynamic, Pooling_leakage, Pooling_area, PreSparsity_dynamic, PreSparsity_leakage,\
	PreSparsity_area, PreSparsity_width, PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area,\
	PostSparsity_width, Scalar_dynamic, Scalar_leakage, Scalar_area, Scalar_width, Transposer_dynamic,\
	Transposer_leakage, Transposer_area, IL, FL, Activation_buffer, Activation_area, Activation_energy,\
	Activation_leakage, Weight_buffer, Weight_area, Weight_energy, Weight_leakage, Mask_buffer, Mask_area,\
	Mask_energy, Mask_leakage, activation_sparsity, weight_sparsity, overlap_factor, data_size, cube_size,\
	matrix_size, DRAM_bandwidth, MainMem_energy, MainMem_leakage):
		self.MacLane = Hardware.MacLane(MacLane_dynamic, MacLane_leakage, MacLane_area, Pmac, activation_sparsity, weight_sparsity, overlap_factor)
		self.DataFlow = Hardware.DataFlow(DataFlow_dynamic, DataFlow_leakage, DataFlow_area)
		self.DMA = Hardware.DMA(DMA_dynamic, DMA_leakage, DMA_area, DMA_bandwidth)
		self.FIFO = Hardware.FIFO(FIFO_dynamic, FIFO_leakage, FIFO_area, FIFO_depth)
		self.BatchNorm = Hardware.BatchNorm(BatchNorm_dynamic, BatchNorm_leakage, BatchNorm_area, BatchNorm_width)
		self.Im2Col = Hardware.Im2Col(Im2Col_dynamic, Im2Col_leakage, Im2Col_area)
		self.Loss = Hardware.Loss(Loss_dynamic, Loss_leakage, Loss_area, Loss_width)
		self.Pooling = Hardware.Pooling(Pooling_dynamic, Pooling_leakage, Pooling_area)
		self.PreSparsity = Hardware.PreSparsity(PreSparsity_dynamic, PreSparsity_leakage, PreSparsity_area, PreSparsity_width)
		self.PostSparsity = Hardware.PostSparsity(PostSparsity_dynamic, PostSparsity_leakage, PostSparsity_area, PostSparsity_width)
		self.Scalar = Hardware.Scalar(Scalar_dynamic, Scalar_leakage, Scalar_area, Scalar_width)
		self.Transposer = Hardware.Transposer(Transposer_dynamic, Transposer_leakage, Transposer_area)
		self.ActivationBuffer = Hardware.Buffer('Activation', int(Activation_buffer * 8 * 1024 * 1024 / (IL + FL)), Activation_area, Activation_energy, Activation_leakage, data_size, IL, FL, activation_sparsity, weight_sparsity, cube_size, matrix_size, DRAM_bandwidth)
		self.WeightBuffer = Hardware.Buffer('Weight', int(Weight_buffer * 8 * 1024 * 1024 / (IL + FL)), Weight_area, Weight_energy, Weight_leakage, data_size, IL, FL, activation_sparsity, weight_sparsity, cube_size, matrix_size, DRAM_bandwidth)
		self.MaskBuffer = Hardware.Buffer('Mask', int(Mask_buffer * 8 * 1024 * 1024 / (IL + FL)), Mask_area, Mask_energy, Mask_leakage, data_size, IL, FL, activation_sparsity, weight_sparsity, cube_size, matrix_size, DRAM_bandwidth)
		self.MainMem_energy = MainMem_energy
		self.MainMem_leakage = MainMem_leakage

		self.InputMemoryLoadQueue = IssueQueue.IssueQueue('MemoryLoad')
		self.WeightMemoryLoadQueue = IssueQueue.IssueQueue('MemoryLoad')
		self.Conv2DQueue = IssueQueue.IssueQueue('Conv2D')
		self.MatMulQueue = IssueQueue.IssueQueue('MatMul')
		self.ScalarQueue = IssueQueue.IssueQueue('Scalar')
		self.BatchNormQueue = IssueQueue.IssueQueue('BatchNorm')
		self.ReluQueue = IssueQueue.IssueQueue('Relu')
		self.PoolingQueue = IssueQueue.IssueQueue('Pooling')
		self.LossQueue = IssueQueue.IssueQueue('Loss')
		self.TransposeQueue = IssueQueue.IssueQueue('Transpose')
		self.InstantQueue = IssueQueue.IssueQueue('Instant')

		self.cycle = 0
		self.dynamic_energy = 0.0
		self.leakage_energy = 0.0
		self.area = self.MacLane.area + self.DataFlow.area + self.DMA.area + self.FIFO.area + self.BatchNorm.area\
		+ self.Im2Col.area + self.Loss.area + self.Pooling.area + self.PreSparsity.area + self.PostSparsity.area\
		+ self.Scalar.area + self.Transposer.area 
		self.area *= 1e-6
		self.area += self.ActivationBuffer.area + self.WeightBuffer.area + self.MaskBuffer.area

		self.leakage_power = self.MacLane.leakage_power + self.DataFlow.leakage_power + self.DMA.leakage_power + self.FIFO.leakage_power + self.BatchNorm.leakage_power\
		+ self.Im2Col.leakage_power + self.Loss.leakage_power + self.Pooling.leakage_power + self.PreSparsity.leakage_power + self.PostSparsity.leakage_power\
		+ self.Scalar.leakage_power + self.Transposer.leakage_power + self.ActivationBuffer.leakage_power + self.WeightBuffer.leakage_power + self.MaskBuffer.leakage_power\
		+ self.MainMem_leakage

	def FillQueue(self, input_pb, Tile, batch_size):
		stream = pb2blocks.Pb2Ops(input_pb)
		stream, tensor_dict, variable_names = pb2blocks.RemoveRedundantOps(stream)
		block_dict = {}
		for op in stream:
			block_dict = pb2blocks.Op2Blocks(block_dict, op, Tile, batch_size)

		for key in block_dict:
			for block in block_dict[key]:
				if type(block) == Blocks.MemoryLoadBlock and block.data_type == 'input':
					self.InputMemoryLoadQueue.Enque(block)
				elif type(block) == Blocks.MemoryLoadBlock and block.data_type == 'weight':
					self.WeightMemoryLoadQueue.Enque(block)
				elif type(block) == Blocks.Conv2DBlock:
					self.Conv2DQueue.Enque(block)
				elif type(block) == Blocks.MatMulBlock:
					self.MatMulQueue.Enque(block)
				elif type(block) == Blocks.ScalarBlock:
					self.ScalarQueue.Enque(block)
				elif type(block) == Blocks.BatchNormBlock:
					self.BatchNormQueue.Enque(block)
				elif type(block) == Blocks.ReluBlock:
					self.ReluQueue.Enque(block)
				elif type(block) == Blocks.PoolingBlock:
					self.PoolingQueue.Enque(block)
				elif type(block) == Blocks.LossBlock:
					self.LossQueue.Enque(block)
				elif type(block) == Blocks.TransposeBlock:
					self.TransposeQueue.Enque(block)
				elif type(block) == Blocks.InstantBlock:
					self.InstantQueue.Enque(block)

	def FillBlocks(self, blocks):
		for block in blocks:
			if type(block) == Blocks.MemoryLoadBlock and block.data_type == 'input':
				self.InputMemoryLoadQueue.Enque(block)
			elif type(block) == Blocks.MemoryLoadBlock and block.data_type == 'weight':
				self.WeightMemoryLoadQueue.Enque(block)
			elif type(block) == Blocks.Conv2DBlock:
				self.Conv2DQueue.Enque(block)
			elif type(block) == Blocks.MatMulBlock:
				self.MatMulQueue.Enque(block)
			elif type(block) == Blocks.ScalarBlock:
				self.ScalarQueue.Enque(block)
			elif type(block) == Blocks.BatchNormBlock:
				self.BatchNormQueue.Enque(block)
			elif type(block) == Blocks.ReluBlock:
				self.ReluQueue.Enque(block)
			elif type(block) == Blocks.PoolingBlock:
				self.PoolingQueue.Enque(block)
			elif type(block) == Blocks.LossBlock:
				self.LossQueue.Enque(block)
			elif type(block) == Blocks.TransposeBlock:
				self.TransposeQueue.Enque(block)
			elif type(block) == Blocks.InstantBlock:
				self.InstantQueue.Enque(block)

	# Assuming only Conv2D, MatMul, and MemoryLoad blocks
	# contribute to the total execution time
	# rest of the blocks will be processed embedded in the pipeline
	def PreProcess(self, clk):
		Scalar_cycle = 0
		BatchNorm_cycle = 0
		Loss_cycle = 0
		while bool(self.ScalarQueue.blocks):
			block = self.ScalarQueue.Issue()
			block.done = True
			cycle = int((block.num + self.Scalar.width - 1) / self.Scalar.width)
			self.dynamic_energy += cycle * 1e-6 / clk * self.Scalar.dynamic_power * 1e-3
			self.cycle += cycle
			Scalar_cycle += cycle

		while bool(self.BatchNormQueue.blocks):
			block = self.BatchNormQueue.Issue()
			block.done = True
			cycle = int((block.num + self.BatchNorm.width - 1) / self.BatchNorm.width)
			self.dynamic_energy += cycle * 1e-6 / clk * self.BatchNorm.dynamic_power * 1e-3
			self.cycle += cycle
			BatchNorm_cycle += cycle

		while bool(self.ReluQueue.blocks):
			block = self.ReluQueue.Issue()
			block.done = True

		while bool(self.PoolingQueue.blocks):
			block = self.PoolingQueue.Issue()
			block.done = True
			self.dynamic_energy += 1e-6 / clk * self.Pooling.dynamic_power * 1e-3

		while bool(self.LossQueue.blocks):
			block = self.LossQueue.Issue()
			block.done = True
			cycle = int((block.num + self.Loss.width - 1) / self.Loss.width)
			self.dynamic_energy += cycle * 1e-6 / clk * self.Loss.dynamic_power * 1e-3
			self.cycle += cycle
			Loss_cycle += cycle

		while bool(self.TransposeQueue.blocks):
			block = self.TransposeQueue.Issue()
			block.done = True
			self.dynamic_energy += 1e-6 / clk * self.Transposer.dynamic_power * 1e-3

		while bool(self.InstantQueue.blocks):
			block = self.InstantQueue.Issue()
			block.done = True

		return Scalar_cycle, BatchNorm_cycle, Loss_cycle


	def Process(self, clk):
		Conv2D_cycle = -1
		MatMul_cycle = -1
		Conv2D_block = None
		MatMul_block = None

		ActivationBufferCount = 0	# counter to record the self.ActivationBuffer.remaining_cycle
		WeightBufferCount = 0		# counter to record the self.WeightBuffer.remaining_cycle
		Memory_cycle = 0

		while bool(self.InputMemoryLoadQueue.blocks) or bool(self.WeightMemoryLoadQueue.blocks):
			print('This is the first while loop.')
			print('Cycle:', self.cycle, 'InputMemoryLoadQueue:', len(self.InputMemoryLoadQueue.blocks), 'WeightMemoryLoadQueue:', len(self.WeightMemoryLoadQueue.blocks),\
			'Conv2DQueue:', len(self.Conv2DQueue.blocks), 'MatMulQueue:', len(self.MatMulQueue.blocks),\
			'ActivationBuffer:', self.ActivationBuffer.remaining_cycle, 'WeightBuffer:', self.WeightBuffer.remaining_cycle, 'MacLane:', self.MacLane.remaining_cycle,\
			'Conv2D_cycle:', Conv2D_cycle, 'MatMul_cycle:', MatMul_cycle)
			# print('Act:', self.ActivationBuffer.data.keys())
			# print('Wgt:', self.WeightBuffer.data.keys())
			
			if bool(self.InputMemoryLoadQueue.blocks):
				if self.ActivationBuffer.ready:					# enter this calculation when self.ActivationBuffer.remaining_cycle = 0
					block = self.InputMemoryLoadQueue.Issue()	# issue the next block in the LoadQueue
					self.ActivationBuffer.Process(block)		# assign data in the buffer
					self.dynamic_energy += 1e-6 / clk * self.ActivationBuffer.remaining_cycle * (self.DMA.dynamic_power + self.MainMem_energy)* 1e-3 # this value scales with f, but it's small from the first place
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9 # this value doesn't change with f, and it's bigger than the above value
					block.done = True

			if bool(self.WeightMemoryLoadQueue.blocks):
				if self.WeightBuffer.ready:						# enter this calculation when self.WeightBuffer.remaining_cycle = 0
					block = self.WeightMemoryLoadQueue.Issue()	# issue the next block in the LoadQueue
					self.WeightBuffer.Process(block)          	# assign data in the buffer
					self.dynamic_energy += 1e-6 / clk * self.WeightBuffer.remaining_cycle * (self.DMA.dynamic_power + self.MainMem_energy) * 1e-3 # this value scales with f, but it's small from the first place
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9 # this value doesn't change with f, and it's bigger than the above value
					block.done = True
            # overall, changing f doesn't change much here

			# in each cycle, self.ActivationBuffer.remaining_cycle -= 1 while self.cycle += 1
			# self.ActivationBuffer.ready = True when self.ActivationBuffer.remaining_cycle = 0
			# the same for WeightBuffer
			# accumulate the remaining_cycles to the counters respectively and set the Buffer.ready directly to skip the non-calculating cycles and speed up
			ActivationBufferCount += self.ActivationBuffer.remaining_cycle
			WeightBufferCount += self.WeightBuffer.remaining_cycle
			self.ActivationBuffer.remaining_cycle = 0
			self.ActivationBuffer.ready = True
			self.WeightBuffer.remaining_cycle = 0
			self.WeightBuffer.ready = True
	
		self.cycle += max(ActivationBufferCount, WeightBufferCount) # add up the max of the accumulated counters to the total self.cycle
		Memory_cycle = max(ActivationBufferCount, WeightBufferCount)

		print('\nDynamic energy:', self.dynamic_energy) 			# check result for the first while loop
		print('Current cycle:', self.cycle, '\n') 					# check result for the first while loop

		ActivationBufferCount = 0	# counter to record the self.ActivationBuffer.remaining_cycle
		WeightBufferCount = 0		# counter to record the self.WeightBuffer.remaining_cycle
		Conv2DCount = 0		# counter to record the Conv2D_cycle
		MatMulCount = 0		# counter to record the MatMul_cycle
		MacLaneCount = 0	# counter to record the MacLane_cycle
		ConvMat_cycle = 0
		
		while 1:
			print('This is the second while loop.')
			print('Cycle:', self.cycle, 'InputMemoryLoadQueue:', len(self.InputMemoryLoadQueue.blocks), 'WeightMemoryLoadQueue:', len(self.WeightMemoryLoadQueue.blocks),\
			'Conv2DQueue:', len(self.Conv2DQueue.blocks), 'MatMulQueue:', len(self.MatMulQueue.blocks),\
			'ActivationBuffer:', self.ActivationBuffer.remaining_cycle, 'WeightBuffer:', self.WeightBuffer.remaining_cycle, 'MacLane:', self.MacLane.remaining_cycle,\
			'Conv2D_cycle:', Conv2D_cycle, 'MatMul_cycle:', MatMul_cycle)

			if bool(self.InputMemoryLoadQueue.blocks):
				if self.ActivationBuffer.ready:
					block = self.InputMemoryLoadQueue.Issue()
					self.ActivationBuffer.Process(block)
					self.dynamic_energy += 1e-6 / clk * self.ActivationBuffer.remaining_cycle * (self.DMA.dynamic_power + self.MainMem_energy)* 1e-3
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9
					block.done = True

			if bool(self.WeightMemoryLoadQueue.blocks):
				if self.WeightBuffer.ready:
					block = self.WeightMemoryLoadQueue.Issue()
					self.WeightBuffer.Process(block)
					self.dynamic_energy += 1e-6 / clk * self.WeightBuffer.remaining_cycle * (self.DMA.dynamic_power + self.MainMem_energy)* 1e-3
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9
					block.done = True

			ActivationBufferCount += self.ActivationBuffer.remaining_cycle
			WeightBufferCount += self.WeightBuffer.remaining_cycle
			self.ActivationBuffer.remaining_cycle = 0
			self.ActivationBuffer.ready = True
			self.WeightBuffer.remaining_cycle = 0
			self.WeightBuffer.ready = True
			# above are the same as in the first while loop

			# in each cycle, Conv2D_cycle -= 1 and self.MacLane.remaining_cycle -= 1 while self.cycle += 1
			# self.MacLane.ready = True when self.MacLane.remaining_cycle = 0
			# enter the calculation only when Conv2D_cycle = 0 and self.MacLane.remaining_cycle = 0
			# accumulate Conv2D_cycle and self.MacLane.remaining_cycle and skip the non-calculating loops to speed up 
			if bool(self.Conv2DQueue.blocks):
				if Conv2D_cycle == -1:
					Conv2D_cycle, Conv2D_activation, Conv2D_weight = self.Conv2DQueue.RemainingCycle(self.ActivationBuffer, self.WeightBuffer)  # Conv2D_cycle scales with f
					if Conv2D_activation != None:
						self.InputMemoryLoadQueue.Insert(Conv2D_activation)
					if Conv2D_weight != None:
						self.WeightMemoryLoadQueue.Insert(Conv2D_weight)
					# accumulate the self.MacLane.remaining_cycle and Conv2D_cycle and set self.MacLane.ready directly to skip the non-calculating loops
					Conv2DCount += Conv2D_cycle # + 1
					Conv2D_cycle = 0

				# enter the calculation in each cycle
				if Conv2D_cycle == 0 and self.MacLane.ready:
					# print('Issue')
					Conv2D_block = self.Conv2DQueue.Issue()
					Conv2D_block.done = True
					Conv2D_cycle = -1
					self.MacLane.Compute(Conv2D_block)
					# print('Compute done')
					self.dynamic_energy += 1e-6 / clk * self.MacLane.remaining_cycle * self.MacLane.dynamic_power * 1e-3 # MacLane_cycle doesn't scale with f, so this value doesn't scale with f
					Conv2D_block = None
					self.dynamic_energy += 1e-6 / clk * self.MacLane.remaining_cycle *\
					(self.DataFlow.dynamic_power + self.FIFO.dynamic_power + self.Im2Col.dynamic_power * 2 + self.PreSparsity.dynamic_power\
					+ self.PostSparsity.dynamic_power + self.Transposer.dynamic_power) * 1e-3
					# this value is dominant by Im2Col.dynamic_power, which doesn't scale with f, so overall this value would inversely scale with f

			# the same idea as above
			elif bool(self.MatMulQueue.blocks):
				if MatMul_cycle == -1:
					MatMul_cycle, MatMul_activation, MatMul_weight = self.MatMulQueue.RemainingCycle(self.ActivationBuffer, self.WeightBuffer)
					if MatMul_activation != None:
						self.InputMemoryLoadQueue.Insert(MatMul_activation)
					if MatMul_weight != None:
						self.WeightMemoryLoadQueue.Insert(MatMul_weight)
					MatMulCount += MatMul_cycle # + 1
					MatMul_cycle = 0

				if MatMul_cycle == 0 and self.MacLane.ready:
					MatMul_block = self.MatMulQueue.Issue()
					MatMul_block.done = True
					MatMul_cycle = -1
					self.MacLane.Compute(MatMul_block)
					self.dynamic_energy += 1e-6 / clk * self.MacLane.remaining_cycle * self.MacLane.dynamic_power * 1e-3
					MatMul_block = None
					self.dynamic_energy += 1e-6 / clk * self.MacLane.remaining_cycle *\
					(self.DataFlow.dynamic_power + self.FIFO.dynamic_power + self.PreSparsity.dynamic_power + self.PostSparsity.dynamic_power) * 1e-3

			# self.MacLane.Process()
			MacLaneCount += self.MacLane.remaining_cycle
			self.MacLane.remaining_cycle = 0	# set the remaining_cycle to 0
			self.MacLane.ready = True			# set self.MacLane.ready to skip calling the function self.MacLance.Process()

			if not bool(self.InputMemoryLoadQueue.blocks) and not bool(self.WeightMemoryLoadQueue.blocks) and not bool(self.Conv2DQueue.blocks) and not bool(self.MatMulQueue.blocks):
				break

		# add up the max of all the accumulated counters to the total self.cycle
		self.cycle += max(ActivationBufferCount, WeightBufferCount, Conv2DCount, MatMulCount, MacLaneCount)
		ConvMat_cycle = max(ActivationBufferCount, WeightBufferCount, Conv2DCount, MatMulCount, MacLaneCount)
		self.dynamic_energy += (self.ActivationBuffer.total_energy + self.WeightBuffer.total_energy) * 1e-9
		self.leakage_energy += 1e-6 / clk * self.cycle * self.leakage_power * 1e-3
		return Memory_cycle, ConvMat_cycle

	def Print(self, clk):
		total_cycles = self.cycle
		latency = 1e-6 / clk * self.cycle
		area = self.area
		dynamic_energy = self.dynamic_energy
		leakage_energy = self.leakage_energy 
		return total_cycles, latency, area, dynamic_energy, leakage_energy
