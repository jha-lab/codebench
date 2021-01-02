# Test script to generates graphs
# Author :  Shikhar Tuli

import sys

if '../' not in sys.path:
	sys.path.append('../')

# Do not show warnings of deprecated functions
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

from absl import flags
from absl import app

from cnnbench.scripts.generate_graphs import main as graph_generator

FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
FLAGS.max_vertices = 2

# Parse flags before access
FLAGS(sys.argv)

if not os.path.exists(f'../results/vertices_{FLAGS.max_vertices}'):
    os.makedirs(f'../results/vertices_{FLAGS.max_vertices}')

FLAGS.output_file = f'../results/vertices_{FLAGS.max_vertices}/generated_graphs.json'

# Generate graphs
app.run(graph_generator)

