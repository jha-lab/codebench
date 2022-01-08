# import math
# import defines
import Blocks
from math import ceil

class Module(object):
	def __init__(self, device_type, dynamic_power, leakage_power, area):
		self.device_type = device_type
		self.dynamic_power = dynamic_power
		self.leakage_power = leakage_power
		self.area = area
		self.block = None
		self.remaining_cycle = 0
		self.ready = True

	def Process(self):
		if self.remaining_cycle > 0:
			self.remaining_cycle -= 1
			if self.remaining_cycle == 0:
				self.ready = True

class MacLane(Module):
	def __init__(self, dynamic_power, leakage_power, area, Pmac, activation_sparsity, weight_sparsity, overlap_factor):
		Module.__init__(self, 'MacLane', dynamic_power, leakage_power, area)
		self.Pmac = Pmac
		self.activation_sparsity = activation_sparsity
		self.weight_sparsity = weight_sparsity
		self.overlap_factor = overlap_factor
		print(self.Pmac)

	def Compute(self, block):
		if type(block) == Blocks.Conv2DBlock:
			T_input = block.inputs[0]
			T_filter = block.inputs[1]
			T_output = block.outputs
			# S is already considered
			Tib = T_output[1] - T_output[0] + 1
			Tix = T_output[3] - T_output[2] + 1
			Tiy = T_output[5] - T_output[4] + 1
			Tif = T_input[7] - T_input[6] + 1
			Tof = T_output[7] - T_output[6] + 1
			Tkx = T_filter[3] - T_filter[2] + 1
			Tky = T_filter[5] - T_filter[4] + 1
			Pib = self.Pmac[0]
			Pix = self.Pmac[1]
			Piy = self.Pmac[2]
			Pif = self.Pmac[3]
			Pof = self.Pmac[4]
			Pkx = self.Pmac[5]
			Pky = self.Pmac[6]

			self.remaining_cycle = int((Tib + Pib - 1) / Pib) * int((Tix + Pix - 1) / Pix) * int((Tiy + Piy - 1) / Piy)\
			* int((Tif + Pif - 1) / Pif) * int((Tof + Pof - 1) / Pof) * int((Tkx + Pkx - 1) / Pkx) * int((Tky + Pky - 1) / Pky)
			self.remaining_cycle = ceil(self.remaining_cycle * (1.0 - self.activation_sparsity) * (1.0 - self.weight_sparsity) * self.overlap_factor)
			self.ready = False
		elif type(block) == Blocks.MatMulBlock:
			mac = self.Pmac[0] * self.Pmac[1] * self.Pmac[2] * self.Pmac[3] * self.Pmac[4] * self.Pmac[5] * self.Pmac[6]
			self.remaining_cycle = int((block.inputs[0] * block.inputs[1] * block.inputs[2] + mac - 1) / mac)
			self.remaining_cycle = ceil(self.remaining_cycle * (1.0 - self.activation_sparsity) * (1.0 - self.weight_sparsity) * self.overlap_factor)
			self.ready = False

class DataFlow(Module):
	def __init__(self, dynamic_power, leakage_power, area):
		Module.__init__(self, 'DataFlow', dynamic_power, leakage_power, area)

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class DMA(Module):
	def __init__(self, dynamic_power, leakage_power, area, bandwidth):
		Module.__init__(self, 'DMA', dynamic_power, leakage_power, area)
		self.bandwidth = bandwidth

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class FIFO(Module):
	def __init__(self, dynamic_power, leakage_power, area, depth):
		Module.__init__(self, 'FIFO', dynamic_power, leakage_power, area)
		self.depth = depth

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class BatchNorm(Module):
	def __init__(self, dynamic_power, leakage_power, area, width):
		Module.__init__(self, 'BatchNorm', dynamic_power, leakage_power, area)
		self.width = width

	def Compute(self, block):
		self.remaining_cycle = int((block.num + self.width - 1) / self.width)
		self.ready = False

class Im2Col(Module):
	def __init__(self, dynamic_power, leakage_power, area):
		Module.__init__(self, 'Im2Col', dynamic_power, leakage_power, area)

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

'''
class Loss(Module):
	def __init__(self, dynamic_power, leakage_power, area, width):
		Module.__init__(self, 'Loss', dynamic_power, leakage_power, area)
		self.width = width

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False
'''

class Pooling(Module):
	def __init__(self, dynamic_power, leakage_power, area, depth):
		Module.__init__(self, 'Pooling', dynamic_power, leakage_power, area)
		self.depth = depth

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class GlobalAvgPooling(Module):
	def __init__(self, dynamic_power, leakage_power, area, depth):
		Module.__init__(self, 'GlobalAvgPooling', dynamic_power, leakage_power, area)
		self.depth = depth

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class Upsampling(Module):
	def __init__(self, dynamic_power, leakage_power, area, depth):
		Module.__init__(self, 'Upsampling', dynamic_power, leakage_power, area)
		self.depth = depth

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class PreSparsity(Module):
	def __init__(self, dynamic_power, leakage_power, area, width):
		Module.__init__(self, 'PreSparsity', dynamic_power, leakage_power, area)
		self.width = width

	def Compute(block):
		self.remaining_cycle = 1
		self.ready = False

class PostSparsity(Module):
	def __init__(self, dynamic_power, leakage_power, area, width):
		Module.__init__(self, 'PostSparsity', dynamic_power, leakage_power, area)
		self.width = width

	def Compute(self, block):
		self.remaining_cycle = 1
		self.ready = False

class Data(object):
	def __init__(self, name, index):
		self.name = name
		self.index = index

class Buffer(object):
	def __init__(self, buffer_type, buffer_size, area, access_energy, leakage_power, data_size, IL, FL, activation_sparsity, weight_sparsity, cube_size, matrix_size, DRAM_bandwidth):
		self.buffer_type = buffer_type
		self.buffer_size = buffer_size
		self.data = {}
		self.current = 0
		self.area = area
		self.access_energy = access_energy
		self.leakage_power = leakage_power
		self.remaining_cycle = 0
		self.ready = True
		self.total_energy = 0.0
		self.block_buffer = None
		self.cycle_buffer = None
		self.data_buffer = None
		self.data_size = data_size
		self.IL = IL
		self.FL = FL
		self.activation_sparsity = activation_sparsity
		self.weight_sparsity = weight_sparsity
		self.cube_size = cube_size
		self.matrix_size = matrix_size
		self.DRAM_bandwidth = DRAM_bandwidth

	def Load(self, data):
		if self.buffer_type == 'Activation':
			self.current += int(self.data_size * (self.IL + self.FL) * (1.0 - self.activation_sparsity))
		elif self.buffer_type == 'Weight':
			self.current += int(self.data_size * (self.IL + self.FL) * (1.0 - self.weight_sparsity))
		self.total_energy += self.access_energy
		if data.name not in self.data:
			self.data[data.name] = []
		self.data[data.name].append(data.index)

	def Store(self, name):
		if self.buffer_type == 'Activation':
			self.current -= int(self.data_size * (self.IL + self.FL) * (1.0 - self.activation_sparsity))
		elif self.buffer_type == 'Weight':
			self.current -= int(self.data_size * (self.IL + self.FL) * (1.0 - self.weight_sparsity))
		self.total_energy += self.access_energy
		del self.data[name][0]
		if self.data[name] == []:
			self.data.pop(name)

	def Check(self, data):
		if data.name not in self.data:
			return False
		elif data.index not in self.data[data.name]:
			return False
		else:
			return True

	def SplitConvData(self, b, x, y, z, block):
		data = Data(block.name, [b, b, x, x + self.cube_size - 1, y, y + self.cube_size - 1, z, z + self.cube_size - 1])
		return data

	def SplitMatData(self, x, y, block):
		data = Data(block.name, [x, x + self.matrix_size - 1, y, y + self.matrix_size - 1])
		return data

	def Split(self, block):
		assert(type(block) is Blocks.MemoryLoadBlock)
		results = []
		if len(block.outputs) == 8:
			results = [self.SplitConvData(b, x, y, z, block) for b in range(block.outputs[0], block.outputs[1] + 1)\
				for x in range(block.outputs[2], block.outputs[3] + 1, self.cube_size) for y in range(block.outputs[4], block.outputs[5] + 1, self.cube_size)\
				for z in range(block.outputs[6], block.outputs[7] + 1, self.cube_size)]
		elif len(block.outputs) == 4:
			results = [self.SplitMatData(x, y, block) for x in range(block.outputs[0], block.outputs[1] + 1, self.matrix_size)\
				for y in range(block.outputs[2], block.outputs[3] + 1, self.matrix_size)]
		return results

	def Process(self, block):
		if block == self.block_buffer:
			self.remaining_cycle = self.cycle_buffer
			if self.remaining_cycle > 0:
				self.ready = False

				if self.buffer_type == 'Activation':
					current = int(self.data_size * (self.IL + self.FL) * (1.0 - self.activation_sparsity))
				elif self.buffer_type == 'Weight':
					current = int(self.data_size * (self.IL + self.FL) * (1.0 - self.weight_sparsity))
				self.current += current * len(self.data_buffer)
				self.total_energy += self.access_energy * len(self.data_buffer)

				for data in self.data_buffer:
					if data.name not in self.data:
						self.data[data.name] = []
					self.data[data.name].append(data.index)

				if self.current > self.buffer_size:
					N = ceil((self.current - self.buffer_size) / current)
					n = N
					while N > 0:
						name = list(self.data.keys())[0]
						if len(self.data[name]) <= N:
							N -= len(self.data[name])
							self.data.pop(name)
						else:
							del self.data[name][:N]
							N = 0
					self.current -= current * n
					self.total_energy += self.access_energy * n
			else:
				self.ready = True
			return
			
		if block == None:
			self.remaining_cycle = 0
			self.ready = True
			return

		if self.buffer_type == 'Activation':
			factor = 1.0 - self.activation_sparsity
		elif self.buffer_type == 'Weight':
			factor = 1.0 - self.weight_sparsity

		results = self.Split(block)
		results = [x for x in results if not self.Check(x)]

		if len(results) > 0:
			self.remaining_cycle = ceil(len(results) * self.data_size * (self.IL + self.FL) * factor / self.DRAM_bandwidth)
			self.ready = False
			for data in results:
				self.Load(data)
			while self.current > self.buffer_size:
				self.Store(list(self.data.keys())[0])
		else:
			self.remaining_cycle = 0
			self.ready = True

	def Remain(self, block):
		if block == None:
			return 0
		if self.buffer_type == 'Activation':
			factor = 1.0 - self.activation_sparsity
		elif self.buffer_type == 'Weight':
			factor = 1.0 - self.weight_sparsity

		results = self.Split(block)
		results = [x for x in results if not self.Check(x)]

		if len(results) > 0:
			self.block_buffer = block
			self.cycle_buffer = ceil(len(results) * self.data_size * (self.IL + self.FL) * factor / self.DRAM_bandwidth)
			self.data_buffer = results
			return self.cycle_buffer
		else:
			return 0
