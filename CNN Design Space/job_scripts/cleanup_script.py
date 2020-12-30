# Cleans up unused directories
# Author :  Shikhar Tuli

import sys

if '../' not in sys.path:
	sys.path.append('../')

# Do not show warnings of deprecated functions
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

import shutil

assert len(sys.argv) == 2, "Takes exactly one argument [model output directory]"

for root, dirnames, filenames in os.walk(sys.argv[1]):
    for dirname in dirnames:
    	if dirname.startswith('eval'):
	        print(f'Deleting directory: {os.path.join(root, dirname)}')
	        shutil.rmtree(os.path.join(root, dirname))
