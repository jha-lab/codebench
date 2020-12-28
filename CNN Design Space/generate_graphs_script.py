# Test script to generates graphs
# Author :  Shikhar Tuli

import sys

if './nasbench-master/' not in sys.path:
	sys.path.append('./nasbench-master/')

from absl import flags
from absl import app

from nasbench.scripts.generate_graphs import main as graph_generator

FLAGS = flags.FLAGS

# Change default flag values
# Those flag values define in command line take precedence
FLAGS.max_vertices = 5

# Parse flags before access
FLAGS(sys.argv)

FLAGS.output_file = f'./results/generated_graphs_vertices_{FLAGS.max_vertices}.json'

# Generate graphs
app.run(graph_generator)

