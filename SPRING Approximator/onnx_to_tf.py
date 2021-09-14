import argparse
import onnx

from onnx_tf.backend import prepare

parser = argparse.ArgumentParser()
parser.add_argument('in_file', type = str, help = 'The input onnx file.')
parser.add_argument('o_file', type = str, help = 'The output pb file.')
args = parser.parse_args()

onnx_model = onnx.load(args.in_file)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(args.o_file)  # export the model
