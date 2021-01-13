import pb2blocks
import Blocks
import IssueQueue
import Hardware
import defines
import numpy as np

class Accelerator(object):
	def __init__(self):
		self.MacLane = Hardware.MacLane(defines.MacLane_dynamic, defines.MacLane_leakage, defines.MacLane_area, defines.Pmac)
		self.DataFlow = Hardware.DataFlow(defines.DataFlow_dynamic, defines.DataFlow_leakage, defines.DataFlow_area)
		self.DMA = Hardware.DMA(defines.DMA_dynamic, defines.DMA_leakage, defines.DMA_area, defines.DMA_bandwidth)
		self.FIFO = Hardware.FIFO(defines.FIFO_dynamic, defines.FIFO_leakage, defines.FIFO_area, defines.FIFO_depth)
		self.BatchNorm = Hardware.BatchNorm(defines.BatchNorm_dynamic, defines.BatchNorm_leakage, defines.BatchNorm_area, defines.BatchNorm_width)
		self.Im2Col = Hardware.Im2Col(defines.Im2Col_dynamic, defines.Im2Col_leakage, defines.Im2Col_area)
		self.Loss = Hardware.Loss(defines.Loss_dynamic, defines.Loss_leakage, defines.Loss_area, defines.Loss_width)
		self.Pooling = Hardware.Pooling(defines.Pooling_dynamic, defines.Pooling_leakage, defines.Pooling_area)
		self.PreSparsity = Hardware.PreSparsity(defines.PreSparsity_dynamic, defines.PreSparsity_leakage, defines.PreSparsity_area, defines.PreSparsity_width)
		self.PostSparsity = Hardware.PostSparsity(defines.PostSparsity_dynamic, defines.PostSparsity_leakage, defines.PostSparsity_area, defines.PostSparsity_width)
		self.Scalar = Hardware.Scalar(defines.Scalar_dynamic, defines.Scalar_leakage, defines.Scalar_area, defines.Scalar_width)
		self.Transposer = Hardware.Transposer(defines.Transposer_area, defines.Transposer_dynamic, defines.Transposer_leakage)
		self.ActivationBuffer = Hardware.Buffer('Activation', int(defines.Activation_buffer * 8 * 1024 * 1024 / (defines.IL + defines.FL)), defines.Activation_area, defines.Activation_energy, defines.Activation_leakage)
		self.WeightBuffer = Hardware.Buffer('Weight', int(defines.Weight_buffer * 8 * 1024 * 1024 / (defines.IL + defines.FL)), defines.Weight_area, defines.Weight_energy, defines.Weight_leakage)
		self.MaskBuffer = Hardware.Buffer('Mask', int(defines.Mask_buffer * 8 * 1024 * 1024 / (defines.IL + defines.FL)), defines.Mask_area, defines.Mask_energy, defines.Mask_leakage)

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
		+ self.Scalar.area + self.Transposer.area + self.ActivationBuffer.area + self.WeightBuffer.area + self.MaskBuffer.area
		self.area *= 1e-6

		self.leakage_power = self.MacLane.leakage_power + self.DataFlow.leakage_power + self.DMA.leakage_power + self.FIFO.leakage_power + self.BatchNorm.leakage_power\
		+ self.Im2Col.leakage_power + self.Loss.leakage_power + self.Pooling.leakage_power + self.PreSparsity.leakage_power + self.PostSparsity.leakage_power\
		+ self.Scalar.leakage_power + self.Transposer.leakage_power + self.ActivationBuffer.leakage_power + self.WeightBuffer.leakage_power + self.MaskBuffer.leakage_power

	def FillQueue(self, input_pb):
		stream = pb2blocks.Pb2Ops(input_pb)
		stream, tensor_dict, variable_names = pb2blocks.RemoveRedundantOps(stream)
		block_dict = {}
		for op in stream:
			block_dict = pb2blocks.Op2Blocks(block_dict, op, defines.Tile, defines.batch_size)

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
	def PreProcess(self):
		while bool(self.ScalarQueue.blocks):
			block = self.ScalarQueue.Issue()
			block.done = True
			cycle = int((block.num + self.Scalar.width - 1) / self.Scalar.width)
			self.dynamic_energy += 1000 / defines.clk * cycle * 1e-9 * self.Scalar.dynamic_power * 1e-3
			self.cycle += cycle

		while bool(self.BatchNormQueue.blocks):
			block = self.BatchNormQueue.Issue()
			block.done = True
			cycle = int((block.num + self.BatchNorm.width - 1) / self.BatchNorm.width)
			self.dynamic_energy += 1000 / defines.clk * cycle * 1e-9 * self.BatchNorm.dynamic_power * 1e-3
			self.cycle += cycle

		while bool(self.ReluQueue.blocks):
			block = self.ReluQueue.Issue()
			block.done = True

		while bool(self.PoolingQueue.blocks):
			block = self.PoolingQueue.Issue()
			block.done = True
			self.dynamic_energy += 1000 / defines.clk * 1e-9 * self.Pooling.dynamic_power * 1e-3

		while bool(self.LossQueue.blocks):
			block = self.LossQueue.Issue()
			block.done = True
			cycle = int((block.num + self.Loss.width - 1) / self.Loss.width)
			self.dynamic_energy += 1000 / defines.clk * cycle * 1e-9 * self.Loss.dynamic_power * 1e-3
			self.cycle += cycle

		while bool(self.TransposeQueue.blocks):
			block = self.TransposeQueue.Issue()
			block.done = True
			self.dynamic_energy += 1000 / defines.clk * 1e-9 * self.Transposer.dynamic_power * 1e-3

		while bool(self.InstantQueue.blocks):
			block = self.InstantQueue.Issue()
			block.done = True


	def Process(self):
		Conv2D_cycle = -1
		MatMul_cycle = -1
		Conv2D_block = None
		MatMul_block = None

		ActivationBufferCount = 0	# counter to record the self.ActivationBuffer.remaining_cycle
		WeightBufferCount = 0		# counter to record the self.WeightBuffer.remaining_cycle

		while bool(self.InputMemoryLoadQueue.blocks) or bool(self.WeightMemoryLoadQueue.blocks):
			print('This is the first while loop.')
			print('Cycle:', self.cycle, 'InputMemoryLoadQueue:', len(self.InputMemoryLoadQueue.blocks), 'WeightMemoryLoadQueue:', len(self.WeightMemoryLoadQueue.blocks),\
			'Conv2DQueue:', len(self.Conv2DQueue.blocks), 'MatMulQueue:', len(self.MatMulQueue.blocks),\
			'ActivationBuffer:', self.ActivationBuffer.remaining_cycle, 'WeightBuffer:', self.WeightBuffer.remaining_cycle, 'MacLane:', self.MacLane.remaining_cycle,\
			'Conv2D_cycle:', Conv2D_cycle, 'MatMul_cycle:', MatMul_cycle)
			
			if bool(self.InputMemoryLoadQueue.blocks):
				if self.ActivationBuffer.ready:					# enter this calculation when self.ActivationBuffer.remaining_cycle = 0
					block = self.InputMemoryLoadQueue.Issue()	# issue the next block in the LoadQueue
					self.ActivationBuffer.Process(block)		# assign data in the buffer
					self.dynamic_energy += 1000 / defines.clk * self.ActivationBuffer.remaining_cycle * 1e-9 * self.DMA.dynamic_power * 1e-3
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9
					block.done = True

			if bool(self.WeightMemoryLoadQueue.blocks):
				if self.WeightBuffer.ready:						# enter this calculation when self.WeightBuffer.remaining_cycle = 0
					block = self.WeightMemoryLoadQueue.Issue()	# issue the next block in the LoadQueue
					self.WeightBuffer.Process(block)          	# assign data in the buffer
					self.dynamic_energy += 1000 / defines.clk * self.WeightBuffer.remaining_cycle * 1e-9 * self.DMA.dynamic_power * 1e-3
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9
					block.done = True

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
		print('\nDynamic energy:', self.dynamic_energy) 			# check result for the first while loop
		print('Current cycle:', self.cycle, '\n') 					# check result for the first while loop

		Conv2DCount = 0		# counter to record the Conv2D_cycle
		MatMulCount = 0		# counter to record the MatMul_cycle
		MacLaneCount = 0	# counter to record the MacLane_cycle
		
		while True:
			print('This is the second while loop.')
			print('Cycle:', self.cycle, 'InputMemoryLoadQueue:', len(self.InputMemoryLoadQueue.blocks), 'WeightMemoryLoadQueue:', len(self.WeightMemoryLoadQueue.blocks),\
			'Conv2DQueue:', len(self.Conv2DQueue.blocks), 'MatMulQueue:', len(self.MatMulQueue.blocks),\
			'ActivationBuffer:', self.ActivationBuffer.remaining_cycle, 'WeightBuffer:', self.WeightBuffer.remaining_cycle, 'MacLane:', self.MacLane.remaining_cycle,\
			'Conv2D_cycle:', Conv2D_cycle, 'MatMul_cycle:', MatMul_cycle)

			if bool(self.InputMemoryLoadQueue.blocks):
				if self.ActivationBuffer.ready:
					block = self.InputMemoryLoadQueue.Issue()
					self.ActivationBuffer.Process(block)
					self.dynamic_energy += 1000 / defines.clk * self.ActivationBuffer.remaining_cycle * 1e-9 * self.DMA.dynamic_power * 1e-3
					self.dynamic_energy += self.MaskBuffer.access_energy * 1e-9
					block.done = True

			if bool(self.WeightMemoryLoadQueue.blocks):
				if self.WeightBuffer.ready:
					block = self.WeightMemoryLoadQueue.Issue()
					self.WeightBuffer.Process(block)
					self.dynamic_energy += 1000 / defines.clk * self.WeightBuffer.remaining_cycle * 1e-9 * self.DMA.dynamic_power * 1e-3
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
					Conv2D_cycle, Conv2D_activation, Conv2D_weight = self.Conv2DQueue.RemainingCycle(self.ActivationBuffer, self.WeightBuffer)
					if Conv2D_activation != None:
						self.InputMemoryLoadQueue.Insert(Conv2D_activation)
					if Conv2D_weight != None:
						self.WeightMemoryLoadQueue.Insert(Conv2D_weight)
					# accumulate the self.MacLane.remaining_cycle and Conv2D_cycle and set self.MacLane.ready directly to skip the non-calculating loops
					MacLaneCount += self.MacLane.remaining_cycle + 2
					self.MacLane.remaining_cycle = 0	# set the remaining_cycle to 0
					self.MacLane.ready = True			# set self.MacLane.ready to skip calling the function self.MacLance.Process()
					Conv2DCount += Conv2D_cycle
					Conv2D_cycle = 0

				# enter the calculation in each cycle
				if Conv2D_cycle == 0 and self.MacLane.ready:
					Conv2D_block = self.Conv2DQueue.Issue()
					Conv2D_block.done = True
					Conv2D_cycle = -1
					self.MacLane.Compute(Conv2D_block)
					self.dynamic_energy += 1000 / defines.clk * self.MacLane.remaining_cycle * 1e-9 * self.MacLane.dynamic_power * 1e-3
					Conv2D_block = None
					self.dynamic_energy += 1000 / defines.clk * self.MacLane.remaining_cycle * 1e-9 *\
					(self.DataFlow.dynamic_power + self.FIFO.dynamic_power + self.Im2Col.dynamic_power * 2 + self.PreSparsity.dynamic_power\
					+ self.PostSparsity.dynamic_power + self.Transposer.dynamic_power) * 1e-3

			# the same idea as above
			elif bool(self.MatMulQueue.blocks):
				if MatMul_cycle == -1:
					MatMul_cycle, MatMul_activation, MatMul_weight = self.MatMulQueue.RemainingCycle(self.ActivationBuffer, self.WeightBuffer)
					if MatMul_activation != None:
						self.InputMemoryLoadQueue.Insert(MatMul_activation)
					if MatMul_weight != None:
						self.WeightMemoryLoadQueue.Insert(MatMul_weight)
					MacLaneCount += self.MacLane.remaining_cycle + 2
					self.MacLane.remaining_cycle = 0	# set the remaining_cycle to 0
					self.MacLane.ready = True       	# set self.MacLane.ready to skip calling the function self.MacLance.Process()
					MatMulCount += MatMul_cycle
					MatMul_cycle = 0

				if MatMul_cycle == 0 and self.MacLane.ready:
					MatMul_block = self.MatMulQueue.Issue()
					MatMul_block.done = True
					MatMul_cycle = -1
					self.MacLane.Compute(MatMul_block)
					self.dynamic_energy += 1000 / defines.clk * self.MacLane.remaining_cycle * 1e-9 * self.MacLane.dynamic_power * 1e-3
					MatMul_block = None
					self.dynamic_energy += 1000 / defines.clk * self.MacLane.remaining_cycle * 1e-9 *\
					(self.DataFlow.dynamic_power + self.FIFO.dynamic_power + self.PreSparsity.dynamic_power + self.PostSparsity.dynamic_power) * 1e-3

			# self.MacLane.Process()

			if not bool(self.InputMemoryLoadQueue.blocks) and not bool(self.WeightMemoryLoadQueue.blocks) and not bool(self.Conv2DQueue.blocks) and not bool(self.MatMulQueue.blocks):
				break

		# add up the max of all the accumulated counters to the total self.cycle
		self.cycle += max(ActivationBufferCount, WeightBufferCount, Conv2DCount, MatMulCount, MacLaneCount)
		self.dynamic_energy += (self.ActivationBuffer.total_energy + self.WeightBuffer.total_energy) * 1e-9
		self.leakage_energy += 1000 / defines.clk * self.cycle * 1e-9 * self.leakage_power * 1e-3

	def Print(self):
		print('Total Cycles: ', self.cycle)
		print('Total execution time: ', 1000 / defines.clk * self.cycle * 1e-9)
		print('Total area: ', self.area)
		print('Dynamic energy consumption: ', self.dynamic_energy)
		print('Leakage energy consumption: ', self.leakage_energy)
