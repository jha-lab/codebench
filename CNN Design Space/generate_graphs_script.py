# Test script to generates graphs
# Author :  Shikhar Tuli

import sys

if './nasbench-master/' not in sys.path:
	sys.path.append('./nasbench-master/')

# Do not show warnings of deprecated functions
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

from absl import flags
from absl import app

from nasbench.scripts.generate_graphs import main as graph_generator

FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
FLAGS.max_vertices = 2

# Parse flags before access
FLAGS(sys.argv)

FLAGS.output_file = f'./results/generated_graphs_vertices_{FLAGS.max_vertices}.json'

# Generate graphs
app.run(graph_generator)

