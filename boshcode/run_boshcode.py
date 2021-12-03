# Run Bayesian Optimization using Second-order gradients and a Heteroscedastic
# surrogate model for Co-Design of CNNs and Accelerators (BOSHCODE)

# Author : Shikhar Tuli


import os
import sys
sys.path.append('../cnn_design-space/cnnbench/')
sys.path.append('../boshnas/boshnas/')

import argparse
import numpy as np
import yaml
import random
import tabulate
import subprocess
import time
import json
import hashlib

import torch

from six.moves import cPickle as pickle

from boshnas_2inp import BOSHNAS as BOSHCODE
from acq import gosh_acq as acq

from library import GraphLib, Graph
from utils import print_util as pu


CONF_INTERVAL = 0.005 # Corresponds to 0.5% accuracy for 95% confidence interval
OVERLAP_THRESHOLD = 0.9 # Corresponds to the minimum overlap for model to be considered

DEBUG = False
PERFORMANCE_PATIENCE = 10 # Convergence criterion for accuracy
ALEATORIC_QUERIES = 10 # Number of queries to be run with aleatoric uncertainty
K = 10 # Number of parallel cold restarts for BOSHNAS
UNC_PROB = 0.1
DIV_PROB = 0.1

MAX_LATENCY = 1 # Maximum latency in seconds
MAX_AREA = 10 # Maximum area in mm^2
MAX_DYNAMIC_ENERGY = 1 # Maximum dynamic energy in Joules
MAX_LEAKAGE_ENERGY = 1 # Maximum leakage energy in Joules


def worker(cnn_config_file: str,
	graphlib_file: str,
	cnn_models_dir: str,
	accel_models_dir: str,
	cnn_model_hash: str,
	chosen_neighbor_hash: str,
	autotune: bool,
	trained_cnn_hashes: list,
	accel_emb: np.array,
	accel_hash: str,
	cluster: str,
	id: str):
	"""Worker to finetune or pretrain the given model
	
	Args:
	    cnn_config_file (str): path to the CNN configuration file
	    graphlib_file (str): path the the graphLib dataset file
	    cnn_models_dir (str): path to the CNN models directory
	    accel_models_dir (str): path to the Accelerators directory
	    cnn_model_hash (str): hash of the given CNN model
	    chosen_neighbor_hash (str): hash of the chosen neighbor
	    autotune (bool): to autotune the given model
	    trained_cnn_hashes (list): list of all CNN hashes that have been trained
	    accel_emb (np.array): embedding of the Accelerator to be simulated
	    accel_hash (str): hash for the given CNN-Accelerator pair
	    cluster (str): name of the cluster - "adroit", "tiger" or "della"
	    id (str): PU-NetID that is used to run slurm commands
	
	Returns:
	    job_id, scratch (str, bool): Job ID for the slurm scheduler and whether CNN model
	    	is being trained from scratch.
	"""
	scratch = False

	print(f'Training CNN model with hash: {cnn_model_hash}.')
	print(f'Simulating on Accelerator with embedding: {accel_emb}.')

	graphLib = GraphLib.load_from_dataset(graphlib_file)

	with open(cnn_config_file) as file:
		try:
			cnn_config = yaml.safe_load(file)
		except yaml.YAMLError as exc:
			raise exc

	chosen_neighbor_path = None
	if chosen_neighbor_hash is not None:
		# Load weights of current model using the finetuned neighbor that was chosen
		chosen_neighbor_path = os.path.join(cnn_models_dir, chosen_neighbor_hash, 'model.pt')
		print(f'Weights copied from neighbor model with hash: {chosen_neighbor_hash}.')
	else:
		scratch = True
		print('No neighbor found for the CNN. Training model from scratch.')

	args = ['--dataset', cnn_config['dataset']]
	args.extend(['--cluster', cluster])
	args.extend(['--id', id])
	args.extend(['--autotune', '1' if autotune else '0'])
	args.extend(['--cnn_model_hash', cnn_model_hash])
	args.extend(['--cnn_model_dir', os.path.join(cnn_models_dir, cnn_model_hash)])
	args.extend(['--cnn_config_file', cnn_config_file])
	args.extend(['--graphlib_file', graphlib_file])
	args.extend(['--train_cnn', '1' if cnn_model_hash not in trained_cnn_hashes else '0'])
	args.extend(['--accel_hash', accel_hash])
	args.extend(['--accel_emb', '\\"' + str(accel_emb).replace('\n', '') + '\\"'])
	args.extend(['--accel_model_file', os.path.join(accel_models_dir, accel_hash) + '.pkl'])

	if chosen_neighbor_path is not None:
		args.extend(['--neighbor_file', chosen_neighbor_path])
	
	slurm_stdout = subprocess.check_output(
		f'ssh della-gpu "cd accelerator_co-design/boshcode; source ./job_scripts/job_worker.sh {" ".join(args)}"',
		shell=True, text=True, executable="/bin/bash")

	return slurm_stdout.split()[-1], scratch
		

def get_job_info(job_id: int):
	"""Obtain job info
	
	Args:
		job_id (int): job id
	
	Returns:
		start_time, elapsed_time, status (str, str, str): job details
	"""
	slurm_stdout = subprocess.check_output(f'ssh della-gpu "slist {job_id}"', shell=True, text=True, executable="/bin/bash")
	slurm_stdout = slurm_stdout.split('\n')[2].split()

	if len(slurm_stdout) > 7:
		start_time, elapsed_time, status = slurm_stdout[5], slurm_stdout[6], slurm_stdout[7]
		if start_time == 'Unknown': start_time = 'UNKNOWN'
	else:
		start_time, elapsed_time, status = 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

	return start_time, elapsed_time, status


def print_jobs(model_jobs: list):
	"""Print summary of all completed, pending and running jobs
	
	Args:
		model_jobs (list): list of jobs
	"""
	header = ['ACCEL HASH', 'JOB ID', 'TRAIN TYPE', 'START TIME', 'ELAPSED TIME', 'STATUS']

	rows = []
	for job in model_jobs:
		start_time, elapsed_time, status = get_job_info(job['job_id'])
		rows.append([job['accel_hash'], job['job_id'], job['train_type'], start_time, elapsed_time, status])

	print()
	print(tabulate.tabulate(rows, header))


def wait_for_jobs(model_jobs: list, running_limit: int = 4, patience: int = 1):
	"""Wait for current jobs in queue to complete
	
	Args:
		model_jobs (list): list of jobs
		running_limit (int, optional): number of running jobs to limit
		patience (int, optional): number of pending jobs to wait for
	"""
	print_jobs(model_jobs)

	completed_jobs = 0
	last_completed_jobs = 0
	running_jobs = np.inf
	pending_jobs = np.inf
	while running_jobs >= running_limit or pending_jobs > patience:
		completed_jobs, running_jobs, pending_jobs = 0, 0, 0
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED': 
				completed_jobs += 1
			elif status == 'PENDING':
				pending_jobs += 1
			elif status == 'RUNNING':
				running_jobs += 1
			elif status == 'FAILED':
				print_jobs(model_jobs)
				raise RuntimeError('Some jobs failed.')
		if last_completed_jobs != completed_jobs:
			print_jobs(model_jobs)
		last_completed_jobs = completed_jobs 
		time.sleep(1)


def update_dataset(graphLib: 'GraphLib', 
	accel_dataset: dict,
	cnn_models_dir: str, 
	accel_models_dir: str, 
	graphlib_file: str, 
	accel_dataset_file: str,
	weights: list):
	"""Update the dataset with all finetuned models
	
	Args:
	    graphLib (GraphLib): GraphLib opject to update
	    accel_dataset (dict): co-design CNN -Accelerator dataset
	    cnn_models_dir (str): directory with all trained CNN models
	    accel_models_dir (str): directory with all simulated CNN-Accelerator pairs
	    graphlib_file (str): path to the graphlib file
	    accel_dataset_file (str): path to the co-design CNN-Accelerator dataset file
	    weights (list): convex combination weights for performance calculation
	
	Returns:
	    best_performance (float): best performance from the current trained CNN models and Accelerators
	"""
	count_cnn = 0
	count_accel = 0
	best_performance = 0

	assert len(weights) == 7, 'The weights list should be of size 7.'
	assert sum(weights) == 1, 'Sum of weights should be equal to 1.'

	# Updating CNN graphLib library 
	for cnn_model_hash in os.listdir(cnn_models_dir):
		checkpoint_path = os.path.join(cnn_models_dir, cnn_model_hash, 'model.pt')
		if os.path.exists(checkpoint_path):
			model_checkpoint = torch.load(checkpoint_path)
			_, model_idx = graphLib.get_graph(model_hash=cnn_model_hash)
			graphLib.library[model_idx].accuracies['train'] = model_checkpoint['train_accuracies'][-1]
			graphLib.library[model_idx].accuracies['val'] = model_checkpoint['val_accuracies'][-1]
			graphLib.library[model_idx].accuracies['test'] = model_checkpoint['test_accuracies'][-1]
			count_cnn += 1

	graphLib.save_dataset(graphlib_file)

	# Updating co-design CNN-Accelerator library
	for accel_hash in accel_dataset.keys():
		accel_trained, cnn_trained = False, False
		if os.path.exists(os.path.join(accel_models_dir, accel_hash) + '.pkl'):
			accel_trained = True
			results = pickle.load(open(os.path.join(accel_models_dir, accel_hash + '.pkl'), 'rb'))
			accel_dataset[accel_hash]['latency'] = results['latency']
			accel_dataset[accel_hash]['area'] = results['area']
			accel_dataset[accel_hash]['dynamic_energy'] = results['dynamic_energy']
			accel_dataset[accel_hash]['leakage_energy'] = results['leakage_energy']
			count_accel += 1
		if os.path.exists(os.path.join(cnn_models_dir, accel_dataset[accel_hash]['cnn_hash'], 'model.pt')):
			cnn_trained = True
			model_checkpoint = torch.load(os.path.join(cnn_models_dir, 
				accel_dataset[accel_hash]['cnn_hash'], 'model.pt'))
			accel_dataset[accel_hash]['train_acc'] = model_checkpoint['train_accuracies'][-1]
			accel_dataset[accel_hash]['val_acc'] = model_checkpoint['val_accuracies'][-1]
			accel_dataset[accel_hash]['test_acc'] = model_checkpoint['test_accuracies'][-1]
		if accel_trained and cnn_trained:
			performance = weights[0] * accel_dataset[accel_hash]['train_acc'] + \
						  weights[1] * accel_dataset[accel_hash]['val_acc'] + \
						  weights[2] * accel_dataset[accel_hash]['test_acc'] + \
						  weights[3] * (1 - accel_dataset[accel_hash]['latency'] / MAX_LATENCY) + \
						  weights[4] * (1 - accel_dataset[accel_hash]['area'] / MAX_AREA) + \
						  weights[5] * (1 - accel_dataset[accel_hash]['dynamic_energy'] / MAX_DYNAMIC_ENERGY) + \
						  weights[6] * (1 - accel_dataset[accel_hash]['leakage_energy'] / MAX_LEAKAGE_ENERGY)
			if performance > best_performance:
				best_performance = performance

	pickle.dump(accel_dataset, open(accel_dataset_file, 'wb+'), pickle.HIGHEST_PROTOCOL)
	print(f'{pu.bcolors.OKGREEN}Co-Design dataset saved to:{pu.bcolors.ENDC} {accel_dataset_file}')

	print()
	print(f'{pu.bcolors.OKGREEN}Trained CNNs in dataset:{pu.bcolors.ENDC} {count_cnn}\n' \
		+ f'{pu.bcolors.OKGREEN}Simulated CNN-Accelerator pairs:{pu.bcolors.ENDC} {count_accel}\n' \
		+ f'{pu.bcolors.OKGREEN}Best performance:{pu.bcolors.ENDC} {best_performance}')
	print()

	return best_performance


def convert_to_tabular(accel_dataset: dict, graphLib: 'GraphLib', weights: list):
	"""Convert the accel_dataset object to a tabular dataset from 
	input encodings to the output loss
	
	Args:
	    accel_dataset (dict): dataset of trained CNN-Accelerator pairs
	    graphLib (GraphLib): GraphLib opject
	    weights (list): convex combination weights for performance calculation
	
	Returns:
	    X_cnn, X_accel, y (tuple): input embeddings and output loss
	"""
	X_cnn, X_accel, y = [], [], []

	assert len(weights) == 7, 'The weights list should be of size 7.'
	assert sum(weights) == 1, 'Sum of weights should be equal to 1.'
	
	for accel_hash in accel_dataset.keys():
		if accel_dataset[accel_hash]['train_acc'] is None or accel_dataset[accel_hash]['latency'] is None: continue
		performance = weights[0] * accel_dataset[accel_hash]['train_acc'] + \
					  weights[1] * accel_dataset[accel_hash]['val_acc'] + \
					  weights[2] * accel_dataset[accel_hash]['test_acc'] + \
					  weights[3] * (1 - accel_dataset[accel_hash]['latency'] / MAX_LATENCY) + \
					  weights[4] * (1 - accel_dataset[accel_hash]['area'] / MAX_AREA) + \
					  weights[5] * (1 - accel_dataset[accel_hash]['dynamic_energy'] / MAX_DYNAMIC_ENERGY) + \
					  weights[6] * (1 - accel_dataset[accel_hash]['leakage_energy'] / MAX_LEAKAGE_ENERGY)
		cnn_graph, _ = graphLib.get_graph(model_hash=accel_dataset[accel_hash]['cnn_hash'])

		X_cnn.append(cnn_graph.embedding)
		X_accel.append(accel_dataset[accel_hash]['accel_emb'])
		y.append(1 - performance)

	X_cnn, X_accel, y = np.array(X_cnn), np.array(X_accel), np.array(y)

	return X_cnn, X_accel, y


def get_neighbor_hash(model: 'Graph', trained_hashes: list):
	chosen_neighbor_hash = None

	# Choose neighbor with max overlap given that it is trained
	for neighbor_hash in model.neighbors:
		if neighbor_hash in trained_hashes: 
			chosen_neighbor_hash = neighbor_hash
			break

	return chosen_neighbor_hash


def main():
	"""Run BOSHCODE to get the best CNN-Accelerator pair in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--fix',
		metavar='',
		type=str,
		help='to fix one part of the search: "cnn" or "accel"',
		default=None)
	parser.add_argument('--index',
		metavar='',
		type=int,
		help='index of the CNN or the Accelerator to fix for one-sided search',
		default=0)
	parser.add_argument('--graphlib_file',
		metavar='',
		type=str,
		help='path to load the CNN graphlib dataset',
		default='../cnn_design-space/cnnbench/dataset/dataset_test.json')
	parser.add_argument('--cnn_config_file',
		metavar='',
		type=str,
		help='path to the CNN configuration file',
		default='../cnn_design-space/cnnbench/configs/CIFAR10/config.yaml')
	parser.add_argument('--accel_embeddings_file',
		metavar='',
		type=str,
		help='path to the accelerator embeddings file',
		default='../accelerator_design-space/accelbench/embeddings/embeddings_test.pkl')
	parser.add_argument('--accel_dataset_file',
		metavar='',
		type=str,
		help='path to the co-design CNN-Accelerator dataset file',
		default='./accel_dataset/accel_dataset_test.pkl')
	parser.add_argument('--surrogate_model_dir',
		metavar='',
		type=str,
		help='path to save the surrogate model parameters',
		default='./surrogate_model')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to the directory where all models are trained',
		default='/scratch/gpfs/stuli/accelerator_co-design')
	parser.add_argument('--performance_weights',
		metavar='',
		type = float,
		nargs = '+',
		help='weights for taking convex combination of different performance values',
		default=[0, 0.2, 0, 0.2, 0.1, 0.2, 0.3])
	parser.add_argument('--num_init',
		metavar='',
		type=int,
		help='number of initial models to initialize the BOSHCODE surrogate model',
		default=10)
	parser.add_argument('--autotune',
		metavar='',
		type=int,
		help='to autotune CNN models or not',
		default=0)
	parser.add_argument('--n_jobs',
		metavar='',
		type=int,
		help='number of parallel jobs for training BOSHCODE',
		default=8)
	parser.add_argument('--cluster',
		metavar='',
		type=str,
		help='name of the cluster - "adroit", "tiger" or "della"',
		default='della')
	parser.add_argument('--id',
		metavar='',
		type=str,
		help='PU-NetID that is used to run slurm commands',
		default='stuli')

	args = parser.parse_args()

	assert args.fix in ['cnn', 'accel', None], '--fix argument should be in ["cnn", "accel", None]'

	random_seed = 0

	# Initialize CNN library
	graphLib = GraphLib.load_from_dataset(args.graphlib_file)

	# Fix the design of CNN embeddings for one-sided search
	if args.fix == 'cnn': 
		print(f'{pu.bcolors.OKBLUE}Fixing CNN to:{pu.bcolors.ENDC}\n{graphLib.library[args.index]}')
		graphLib.library = [graphLib.library[args.index]]

	# Initialize Accelerator embeddings
	accel_embeddings = pickle.load(open(args.accel_embeddings_file, 'rb'))
	accel_embeddings = np.array(accel_embeddings)

	# Fix the design space of Accelerator embeddings for one-sided search
	if args.fix == 'accel': 
		print(f'{pu.bcolors.OKBLUE}Fixing Accelerator to:{pu.bcolors.ENDC}\n{accel_embeddings[args.index, :]}')
		accel_embeddings = accel_embeddings[args.index, :].reshape(1, -1)

	# New dataset file for CNN library
	new_graphlib_file = args.graphlib_file.split('.json')[0] + '_trained.json'

	# Create or load CNN-Accelerator pairs dataset
	accel_dataset = {}
	if os.path.exists(args.accel_dataset_file):
		accel_dataset = pickle.load(open(args.accel_dataset_file, 'rb'))
		if args.fix == 'cnn':
			assert len(set([accel['cnn_hash'] for _, accel in accel_dataset.items()])) == 1, \
				'Stored Co-Design dataset has more than one CNN'
		elif args.fix == 'accel':
			assert len(set([str(accel['accel_emb']) for _, accel in accel_dataset.items()])) == 1, \
				'Stored Co-Design dataset has more than one Accelerator'
	else:
		for accel_idx in range(accel_embeddings.shape[0]):
			for cnn_idx in range(len(graphLib.library)):
				accel_cnn_str = str(accel_embeddings[accel_idx, :]) + graphLib.library[cnn_idx].hash
				accel_dataset[hashlib.sha256(accel_cnn_str.encode('utf-8')).hexdigest()] = \
					{'cnn_hash': graphLib.library[cnn_idx].hash, 'accel_emb': accel_embeddings[accel_idx, :], \
					'train_acc': None, 'val_acc': None, 'test_acc': None, 'latency': None, 'area': None, \
					'dynamic_energy': None, 'leakage_energy': None}
		pickle.dump(accel_dataset, open(args.accel_dataset_file, 'wb+'), pickle.HIGHEST_PROTOCOL)
	accel_hashes = list(accel_dataset.keys())

	# Set autotune for every CNN model trained
	autotune = True if args.autotune == 1 else False
	
	# Initialize a dictionary mapping the CNN or CNN-Accelerator hash to its corresponding job_id
	model_jobs = []

	if not os.path.exists(args.models_dir):
		os.makedirs(args.models_dir)

	cnn_config = yaml.safe_load(open(args.cnn_config_file))

	# Set directories for training of CNN and Accelerator models
	cnn_models_dir = os.path.join(args.models_dir, 'cnnbench_models', cnn_config['dataset'])
	accel_models_dir = os.path.join(args.models_dir, 'accelbench_models')

	# Get trained CNN models and Accelerator architectures
	trained_cnn_hashes = os.listdir(cnn_models_dir)
	trained_accel_hashes = [accel_hash[:-4] for accel_hash in os.listdir(accel_models_dir)]

	# Check trained_accel_hashes have all respective CNNs trained
	trained_accel_hashes_new = []
	for accel_hash in trained_accel_hashes:
		cnn_hash = accel_dataset[accel_hash]['cnn_hash']
		if cnn_hash not in trained_cnn_hashes:
			print(f'CNN-Accelerator pair with hash: {accel_hash}, doesn\'t have respective CNN trained (with hash: {cnn_hash})')
		else:
			trained_accel_hashes_new.append(accel_hash)
	trained_accel_hashes = trained_accel_hashes_new

	# Train randomly sampled models if total trained models is less than num_init
	# TODO: Add skopt.sampler.Sobol points instead
	while len(trained_accel_hashes) < args.num_init:
		sample_idx = random.randint(0, len(accel_hashes)-1)
		accel_hash = accel_hashes[sample_idx]

		if accel_hash not in trained_accel_hashes:
			trained_accel_hashes.append(accel_hash)

			cnn_hash = accel_dataset[accel_hash]['cnn_hash']
			accel_emb = accel_dataset[accel_hash]['accel_emb']

			job_id, scratch = worker(cnn_config_file=args.cnn_config_file, graphlib_file=args.graphlib_file,
				cnn_models_dir=cnn_models_dir, accel_models_dir=accel_models_dir, cnn_model_hash=cnn_hash, 
				chosen_neighbor_hash=None, autotune=autotune, trained_cnn_hashes=trained_cnn_hashes, 
				accel_emb=accel_emb, accel_hash=accel_hash, cluster=args.cluster, id=args.id) 
			assert scratch is True

			train_type = 'S' if scratch else 'WT'
			train_type += ' w/ A' if autotune else ''
			
			model_jobs.append({'accel_hash': accel_hash, 
				'job_id': job_id, 
				'train_type': train_type})

	# Wait for jobs to complete
	wait_for_jobs(model_jobs)

	# Update dataset with newly trained models
	old_best_performance = update_dataset(graphLib, accel_dataset, cnn_models_dir, accel_models_dir, 
		new_graphlib_file, args.accel_dataset_file, args.performance_weights)

	# Get entire dataset in embedding space
	cnn_embeddings = []
	for graph in graphLib.library:
		cnn_embeddings.append(graph.embedding)
	cnn_embeddings = np.array(cnn_embeddings)

	min_cnn, max_cnn = np.min(cnn_embeddings, axis=0), np.max(cnn_embeddings, axis=0)
	min_accel, max_accel = np.min(accel_embeddings, axis=0), np.max(accel_embeddings, axis=0)

	X_ds = []
	for cnn_idx in range(cnn_embeddings.shape[0]):
		for accel_idx in range(accel_embeddings.shape[0]):
			X_ds.append((cnn_embeddings[cnn_idx, :], accel_embeddings[accel_idx, :]))

	# Initialize the two-input BOSHNAS model
	surrogate_model = BOSHCODE(input_dim1=cnn_embeddings.shape[1],
							  input_dim2=accel_embeddings.shape[1],
							  bounds1=(min_cnn, max_cnn),
							  bounds2=(min_accel, max_accel),
							  trust_region=False,
							  second_order=True,
							  parallel=True if not DEBUG else False,
							  model_aleatoric=True,
							  save_path=args.surrogate_model_dir,
							  pretrained=False)

	# Get initial dataset after finetuning num_init models
	X_cnn, X_accel, y = convert_to_tabular(accel_dataset, graphLib, args.performance_weights)
	max_loss = np.amax(y)

	same_performance = 0
	method = 'optimization'

	while same_performance < PERFORMANCE_PATIENCE + ALEATORIC_QUERIES:
		prob = random.uniform(0, 1)
		if 0 <= prob <= (1 - UNC_PROB - DIV_PROB):
			method = 'optimization'
		elif 0 <= prob <= (1 - DIV_PROB):
			method = 'unc_sampling'
		else:
			method = 'div_sampling'

		# Get a set of trained models and models that are currently in the pipeline
		trained_hashes, pipeline_hashes = [], []
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED':
				trained_hashes.append(job['accel_hash'])
			else:
				pipeline_hashes.append(job['accel_hash'])

		new_queries = 0

		if method == 'optimization':
			print(f'{pu.bcolors.OKBLUE}Running optimization step{pu.bcolors.ENDC}')
			# Get current tabular dataset
			X_cnn, X_accel, y = convert_to_tabular(accel_dataset, graphLib, args.performance_weights)
			y = y/max_loss

			# Train BOSHNAS model
			train_error = surrogate_model.train(X_cnn, X_accel, y)

			# Use aleatoric loss close to convergence to optimize training recipe
			if same_performance < PERFORMANCE_PATIENCE:
				# Architecture not converged yet. Use only epistemic uncertainty
				use_al = False
			else:
				# Use aleatoric uncertainty to optimize training recipe
				use_al = True

			# Get next queries
			query_indices = surrogate_model.get_queries(x=X_ds, k=K, explore_type='ucb', use_al=use_al) 

			# Run queries
			for i in set(query_indices):
				accel_hash = accel_hashes[i]
				accel_emb = accel_dataset[accel_hash]['accel_emb']

				if not use_al and accel_hash in trained_accel_hashes + pipeline_hashes:
					# If aleatoric uncertainty is not considered, only consider models that are not 
					# already trained or in the pipeline
					continue

				cnn_model, _ = graphLib.get_graph(model_hash=accel_dataset[accel_hash]['cnn_hash'])
				chosen_neighbor_hash = get_neighbor_hash(cnn_model, trained_cnn_hashes)

				if chosen_neighbor_hash:
					# Finetune model with the chosen neighbor
					job_id, scratch = worker(cnn_config_file=args.cnn_config_file, graphlib_file=args.graphlib_file,
						cnn_models_dir=cnn_models_dir, accel_models_dir=accel_models_dir, cnn_model_hash=cnn_hash, 
						chosen_neighbor_hash=chosen_neighbor_hash, autotune=autotune, trained_cnn_hashes=trained_cnn_hashes, 
						accel_emb=accel_emb, accel_hash=accel_hash, cluster=args.cluster, id=args.id)
					assert scratch is False
				else:
					# If no neighbor was found, proceed to next query model
					continue

				new_queries += 1

				train_type = 'S' if scratch else 'WT'
				train_type += ' w/ A' if autotune else ''
				
				model_jobs.append({'accel_hash': accel_hash, 
					'job_id': job_id, 
					'train_type': train_type})
			
			if new_queries == 0:
				# If no queries were found where weight transfer could be used, train the highest
				# predicted model from scratch
				query_embeddings = [X_ds[idx] for idx in query_indices]
				candidate_predictions = surrogate_model.predict(query_embeddings)

				best_prediction_index = query_indices[np.argmax(acq([pred[0] for pred in candidate_predictions],
																[pred[1][0] for pred in candidate_predictions],
																explore_type='ucb'))]

				accel_hash = accel_hashes[best_prediction_index]
				cnn_model, _ = graphLib.get_graph(model_hash=accel_dataset[accel_hash]['cnn_hash'])

				# Train model
				job_id, scratch = worker(cnn_config_file=args.cnn_config_file, graphlib_file=args.graphlib_file,
					cnn_models_dir=cnn_models_dir, accel_models_dir=accel_models_dir, cnn_model_hash=cnn_hash, 
					chosen_neighbor_hash=None, autotune=autotune, trained_cnn_hashes=trained_cnn_hashes, 
					accel_emb=accel_emb, accel_hash=accel_hash, cluster=args.cluster, id=args.id) 
				assert scratch is True

				train_type = 'S' if scratch else 'WT'
				train_type += ' w/ A' if autotune else ''
				
				model_jobs.append({'accel_hash': accel_hash, 
					'job_id': job_id, 
					'train_type': train_type})

		elif method == 'unc_sampling':
			print(f'{pu.bcolors.OKBLUE}Running uncertainty sampling{pu.bcolors.ENDC}')

			candidate_predictions = surrogate_model.predict(X_ds) 

			# Get model index with highest epistemic uncertainty
			unc_prediction_idx = np.argmax([pred[1][0] for pred in candidate_predictions])

			# Sanity check: CNN-Accelerator pair with highest epistemic uncertainty should not be trained
			assert accel_hashes[unc_prediction_idx] not in trained_accel_hashes

			if accel_hashes[unc_prediction_idx] in pipeline_hashes:
				print(f'{pu.bcolors.OKBLUE}Highest uncertainty model already in pipeline{pu.bcolors.ENDC}')
			else:
				accel_hash = accel_hashes[unc_prediction_idx]
				cnn_model, _ = graphLib.get_graph(model_hash=accel_dataset[accel_hash]['cnn_hash'])
				accel_emb = accel_dataset[accel_hash]['accel_emb']

				# Train model
				job_id, scratch = worker(cnn_config_file=args.cnn_config_file, graphlib_file=args.graphlib_file,
					cnn_models_dir=cnn_models_dir, accel_models_dir=accel_models_dir, cnn_model_hash=cnn_hash, 
					chosen_neighbor_hash=None, autotune=autotune, trained_cnn_hashes=trained_cnn_hashes, 
					accel_emb=accel_emb, accel_hash=accel_hash, cluster=args.cluster, id=args.id) 
				assert scratch is False

				new_queries += 1

				train_type = 'S' if scratch else 'WT'
				train_type += ' w/ A' if autotune else ''
				
				model_jobs.append({'accel_hash': accel_hash, 
					'job_id': job_id, 
					'train_type': train_type})

		else:
			print(f'{pu.bcolors.OKBLUE}Running diversity sampling{pu.bcolors.ENDC}')

			# Get randomly sampled model idx
			# TODO: Add skopt.sampler.Sobol points instead
			unc_prediction_idx = random.randint(0, len(accel_hashes))

			while accel_hashes[unc_prediction_idx] in trained_hashes + pipeline_hashes:
				unc_prediction_idx = random.randint(0, len(accel_hashes))

			accel_hash = accel_hashes[unc_prediction_idx]
			cnn_model, _ = graphLib.get_graph(model_hash=accel_dataset[accel_hash]['cnn_hash'])
			accel_emb = accel_dataset[accel_hash]['accel_emb']

			# Train sampled model
			job_id, scratch = worker(cnn_config_file=args.cnn_config_file, graphlib_file=args.graphlib_file,
				cnn_models_dir=cnn_models_dir, accel_models_dir=accel_models_dir, cnn_model_hash=cnn_hash, 
				chosen_neighbor_hash=None, autotune=autotune, trained_cnn_hashes=trained_cnn_hashes, 
				accel_emb=accel_emb, accel_hash=accel_hash, cluster=args.cluster, id=args.id) 

			new_queries += 1

			train_type = 'S' if scratch else 'WT'
			train_type += ' w/ A' if autotune else ''
			
			model_jobs.append({'accel_hash': accel_hash, 
				'job_id': job_id, 
				'train_type': train_type})

		# Wait for jobs to complete
		wait_for_jobs(model_jobs)

		# Update dataset with newly trained models
		best_performance = update_dataset(graphLib, accel_dataset, cnn_models_dir, accel_models_dir, 
			new_graphlib_file, args.accel_dataset_file, args.performance_weights)

		# Update same_performance to check convergence
		if best_performance == old_best_performance and method == 'optimization':
			same_performance += 1

		old_best_performance = best_performance

	# Wait for jobs to complete
	wait_for_jobs(model_jobs, running_limit=0, patience=0)

	# Update dataset with newly trained models
	best_performance = update_dataset(graphLib, accel_dataset, cnn_models_dir, accel_models_dir, 
		new_graphlib_file, args.accel_dataset_file, args.performance_weights)

	print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()

