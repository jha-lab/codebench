import argparse
import os
import sys
from typing import Iterable
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='The file name of the frozen graph.')
parser.add_argument('output', type=str, help='The file name of the output pickle file that stored operation stream.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

graph_def = None
graph = None

print('Loading graph definition ...', file=sys.stderr)
try:
    with tf.io.gfile.GFile(args.file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
except BaseException as e:
    parser.exit(2, 'Error loading the graph definition: {}'.format(str(e)))

print('Importing graph ...', file=sys.stderr)
try:
    assert graph_def is not None
    with tf.Graph().as_default() as graph:  # type: tf.Graph
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
except BaseException as e:
    parser.exit(2, 'Error importing the graph: {}'.format(str(e)))


assert graph is not None
ops = graph.get_operations()  # type: Iterable[tf.Operation]

stream = []
tensors = []
pending = []

for op in ops:
    if len(op.inputs) == 0:
        stream.append(op)
        for out in op.outputs:
            tensors.append(out)
    else:
        pending.append(op)

while len(pending) > 0:
    print("The number of remaining ops is ", len(pending))
    for op in pending:
        finish = True
        for ins in op.inputs:
            if ins not in tensors:
                finish = False
                break
        if finish:
            stream.append(op)
            for out in op.outputs:
                tensors.append(out)
            pending.remove(op)


