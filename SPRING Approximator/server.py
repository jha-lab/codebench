import Accelerator
import argparse
import os
import time
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='The file name of the pickle object.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

filehandler = open(args.file, 'rb')
blocks = pickle.load(filehandler)

start_time = time.time()

chip = Accelerator.Accelerator()
chip.FillBlocks(blocks)
chip.PreProcess()
chip.Process()
chip.Print()

print('Simulation time:', time.time() - start_time)