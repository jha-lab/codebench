# Fix models in cnnbench_models directory

# Author : Shikhar Tuli



import os
import sys
sys.path.append('../cnn_design-space/cnnbench/')
sys.path.append('../boshnas/boshnas/')

from six.moves import cPickle as pickle
import library
import torch
import shutil

# accel_dataset = pickle.load(open('../boshcode/accel_dataset/accel_dataset_mini.pkl', 'rb'))

cnn_models_dir = './cnnbench_models/CIFAR10/'
accel_models_dir = './accelbench_models'

trained_cnn_hashes = os.listdir(cnn_models_dir)
trained_accel_hashes = [accel_hash[:-4] for accel_hash in os.listdir(accel_models_dir)]

for cnn_hash in trained_cnn_hashes:
	if 'auto_tune' in os.listdir(os.path.join(cnn_models_dir, cnn_hash)):
		print(f'Fixing CNN with hash: {cnn_hash}')

		best_val_accuracy = 0
		best_training_recipe = ''

		for training_recipe in os.listdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune')):
			if os.path.isdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', training_recipe)):
				for ckpt_dir in os.listdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', training_recipe)):
					if ckpt_dir.startswith('checkpoint') and os.path.isdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', training_recipe, ckpt_dir)):
						for ckpt_name in os.listdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', training_recipe, ckpt_dir)):
							if ckpt_name.endswith('.pt'):
								ckpt = torch.load(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', training_recipe, ckpt_dir, ckpt_name))
								if ckpt['val_accuracies'][-1] > best_val_accuracy:
									best_val_accuracy = ckpt['val_accuracies'][-1]
									best_training_recipe = training_recipe

		print(f'\tBest validation accuracy checkpoint: {best_val_accuracy}')

		for ckpt_dir in os.listdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', best_training_recipe)):
			if ckpt_dir.startswith('checkpoint') and os.path.isdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', best_training_recipe, ckpt_dir)):
				for ckpt_name in os.listdir(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', best_training_recipe, ckpt_dir)):
					if ckpt_name.endswith('.pt'):
						ckpt = torch.load(os.path.join(cnn_models_dir, cnn_hash, 'auto_tune', best_training_recipe, ckpt_dir, ckpt_name))
						if ckpt['val_accuracies'][-1] == best_val_accuracy:
							torch.save(ckpt, os.path.join(cnn_models_dir, cnn_hash, 'model.pt'))
