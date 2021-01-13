import Blocks

class IssueQueue(object):
	def __init__(self, block_type):
		self.type = block_type
		self.blocks = []

	def Enque(self, block):
		self.blocks.append(block)

	def Insert(self, block):
		self.blocks.insert(0, block)

	def RemainingCycle(self, activation_buffer=None, weight_buffer=None):
		block = self.blocks[0]
		if self.type == 'MemoryLoad':
			return 0, None, None
		elif self.type == 'Conv2D':
			activation = Blocks.MemoryLoadBlock(block.inputs_name[0], block.number, block.inputs[0], 'input')
			weight = Blocks.MemoryLoadBlock(block.inputs_name[1], block.number, block.inputs[1], 'weight')
			activation_data = activation_buffer.Split(activation)
			weight_data = weight_buffer.Split(weight)

			flag = True
			if not activation_buffer.Check(activation_data[0]):	# approximation: check the first entry to skip the original for loop
				flag = False
			if flag:
				activation = None
			flag = True
			if not weight_buffer.Check(weight_data[0]):			# approximation: check the first entry to skip the original for loop
				flag = False
			if flag:
				weight = None
			
			activation_cycle = activation_buffer.Remain(activation)
			weight_cycle = weight_buffer.Remain(weight)
			return max(activation_cycle, weight_cycle), activation, weight

		elif self.type == 'MatMul':
			activation = Blocks.MemoryLoadBlock(block.inputs_name[0], 0, [0, block.inputs[1] - 1, 0, block.inputs[0] - 1], 'input')
			weight = Blocks.MemoryLoadBlock(block.inputs_name[1], 0, [0, block.inputs[2] - 1, 0, block.inputs[1] - 1], 'weight')
			activation_data = activation_buffer.Split(activation)
			weight_data = weight_buffer.Split(weight)

			flag = True
			if not activation_buffer.Check(activation_data[0]):	# approximation: check the first entry to skip the original for loop
				flag = False
			if flag:
				activation = None
			flag = True
			if not weight_buffer.Check(weight_data[0]):			# approximation: check the first entry to skip the original for loop
				flag = False
			if flag:
				weight = None
			
			activation_cycle = activation_buffer.Remain(activation)
			weight_cycle = weight_buffer.Remain(weight)
			return max(activation_cycle, weight_cycle), activation, weight

	def Issue(self):
		block = self.blocks[0]
		del self.blocks[0]
		return block
