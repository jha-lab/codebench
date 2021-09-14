import pb2blocks
import argparse
import os
import defines
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='The file name of the frozen graph.')
parser.add_argument('output', type=str, help='The file name of the output blocks.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

stream = pb2blocks.Pb2Ops(args.file)
stream, tensor_dict, variable_names = pb2blocks.RemoveRedundantOps(stream)
block_dict = {}
for op in stream:
	block_dict = pb2blocks.Op2Blocks(block_dict, op, defines.Tile, defines.batch_size)

blocks = []
for key in block_dict:
	for block in block_dict[key]:
		blocks.append(block)

filehandler = open(args.output, 'wb')
pickle.dump(blocks, filehandler)
