import argparse
import os
from os import path
import onnx
import onnx2keras
from onnx2keras import onnx_to_keras

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='The path to the onnx model file.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

file_name = path.basename(args.file)
file_name = path.splitext(file_name)[0]

# Load the onnx model and retrieve the input name
onnx_model = onnx.load(args.file)
input_name = onnx_model.graph.input[0].name

# Onnx to Keras
model = onnx_to_keras(onnx_model, [input_name])


# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="yourInputName"))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 50)
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name=file_name + '.pb',
                  as_text=False)
