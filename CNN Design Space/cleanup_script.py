# Generates graphs and trains on multiple workers
# Author :  Shikhar Tuli

import sys

if './nasbench-master/' not in sys.path:
	sys.path.append('./nasbench-master/')

# Do not show warnings of deprecated functions
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL} 

import shutil

f = open('model_dir.txt')
model_dir = f.read()
f.close()

for root, dirnames, filenames in os.walk(model_dir):
    for dirname in dirnames:
    	if dirname.startswith('eval'):
	        print(f'Deleting directory: {os.path.join(root, dirname)}')
	        shutil.rmtree(os.path.join(root, dirname))
