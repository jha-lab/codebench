class Block(object):
	"""class for Block"""
	def __init__(self, block_name, block_type, number=0):
		self.name = block_name
		self.type = block_type
		self.number = number
		self.inputs = None
		self.outputs = None
		self.inputs_name = None
		self.done = False

class MemoryLoadBlock(Block):
	def __init__(self, block_name, number, Tile, data_type):
		Block.__init__(self, block_name, 'MemoryLoad', number)
		self.outputs = Tile
		self.data_type = data_type
		
class Conv2DBlock(Block):
	def __init__(self, block_name, number, inputs, inputs_name, outputs):
		Block.__init__(self, block_name, 'Conv2D', number)
		self.inputs = inputs
		self.inputs_name = inputs_name
		self.outputs = outputs

class MatMulBlock(Block):
	def __init__(self, block_name, inputs, inputs_name):
		Block.__init__(self, block_name, 'MatMulBlock')
		self.inputs = inputs
		self.inputs_name = inputs_name

class ScalarBlock(Block):
	def __init__(self, block_name, scalar_type, num, inputs_name):
		Block.__init__(self, block_name, 'Scalar')
		self.scalar_type = scalar_type
		self.num = num
		self.inputs_name = inputs_name

class BatchNormBlock(object):
	def __init__(self, block_name, num, inputs_name, outputs_name):
		Block.__init__(self, block_name, 'BatchNorm')
		self.num = num
		self.inputs_name = inputs_name
		self.outputs_name = outputs_name

class ReluBlock(object):
	def __init__(self, block_name, num, input_name):
		Block.__init__(self, block_name, 'Relu')
		self.num = num
		self.inputs_name = input_name

class PoolingBlock(object):
	def __init__(self, block_name, pooling_type, number, inputs, input_name, outputs):
		Block.__init__(self, block_name, 'Pooling', number)
		self.pooling_type = pooling_type
		self.inputs = inputs
		self.inputs_name = input_name
		self.outputs = outputs

class LossBlock(object):
	def __init__(self, block_name, loss_type, num, input_name):
		Block.__init__(self, block_name, 'Loss')
		self.loss_type = loss_type
		self.num = num
		self.inputs_name = input_name

class TransposeBlock(object):
	def __init__(self, block_name):
		Block.__init__(self, block_name, 'Transpose')
		
class InstantBlock(object):
	def __init__(self, block_name):
		Block.__init__(self, block_name, 'Instant')
		