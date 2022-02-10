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
import numpy as np

manual_models = {'vgg11': 'ad0a60f4a4077a823e03c806612b1739d7dbdcb9955a77b582d18ba9eea0a6b8',
	'vgg13': '6d0862109b4bd4ba489ae97bc9f675734fc2a6a54a210b3a4f391fa01889a982',
	'vgg16': 'c02c2fde320d179d81af17833c9466bfbfbb1facb42d82e14e9793a9b1ae7979',
	'vgg19': '2bc7895ec217f994d1d1bec034626be9a9f68848c97d62f7e1c493cc8c73330b',
	'resnet18': '1da03cb835aa34f8ccbe0ecf6a025ff9e4f8536e1e1713e9a93c65523b7d885a',
	'resnet34': '0b46c0d8566133b428ec736bea72c8b7eda7e79d477f54c587c1f6a521611998',
	'resnet50': '3abff4667b9e17a30f71ad57a6f7d52d217283984e81a878fceabeb43d299114',
	'resnet101': '4021dbdb0a51fcfab78d87515f07ece1227696d97260b019a76810ba60f5ed03',
	'resnet152': 'e1aff753f15c8284429fe3e6e9a713d6367e151329576d3701967385b17b48e8',
	'shufflenet': '1a135065feaac7541e71be5767980aafb970e67aa6b938a68235192c823bf3ff',
	'mobilenet': '03c30e238db867874e66e985b6fd494d27661cd17f6ceef332415a5af88b647e',
	'googlenet': '7ea48ef10963fb2a794613dfb2e95e5d6c949217fddb17611cb1861951394232',
	'xception': '3227a974145e3074af1fc3c9975b57b6d7b2fa2fa1e95b856f0b5f3b6af32e00'}

manual_accel_hashes = {'vgg11': '08a0734b7c753fd8940a5977dfe968cebfe7f5d0a0d6af13837b7c0623bb4b72',
	'vgg13': 'fa909baf0301c2d43cb80c4935fea98b4a3a13fd97185428fdda12b1d1d0d080',
	'vgg16': '02bd3dacf4b06d0e078d0453d39e501271f350fc01c25f716f92bb2bd2f7f2b5',
	'vgg19': '19bf3b9956047868d1b5546f69595d16b86f440778b28908ab2a61e16849859c',
	'resnet18': 'd8538572e9c68de1d836917553c56899474ed7b2a678094a1b80536f384ed60b',
	'resnet34': '339a5e3d9d2e22ef9c88b1f39df246598c457684643fd6ce8cdd1db5df4b83ea',
	'resnet50': '27061773536be6deea1e3e9a4e6eb4f19331b87548de395ac0fa1b7fbef571ee',
	'resnet101': '05757f512c3da8cb7188c7faf010668ac94663099b72cb22de1667ed54294ca2',
	'resnet152': '33e627ae0e10b466a3732c9411b91cf3ed956cdedde6784443dd6940dff6220c',
	'shufflenet': '774626463f634f909e566043957474d1f2b5eb5aaaa6482c9c123787710cd950',
	'mobilenet': '63bd29899005bdd1f2bce7c1f8f67588408315a9ef5c39b12d000419376b973d',
	'googlenet': 'de8133c28933027135b4167ce4e7c77ae6352064e33d656dd2547556cbf9f6b2',
	'xception': 'ef2574e01ab535e73c7176fd67ef38e3e47c74a91ab39f468b91d0bff6053c58'}

mobilenet_accel_hashes = {'[ 1  1  1  6  4  1  1  1 14  8  1  2  3]': '7b8dd7df200e3f6f55e98a477999e9533a45ded15f09360ee6fdced974717e2d',
	'[ 4 16  7  6  1  1  1 64 10  8  2  1  1]': '07a2b32e9eb0a8b9a555f43bc8d9a28f7be80169547a2c7b91a59831f652f880',
	'[ 2  1  2  7  2  5  5  1 12 14  2  1  6]': '0ecf32f36983a4ea3ad47b258a7d37d0a5bbf453401084aa9fb6123e75e8ce57',
	'[  4   1   4   3   4   5   5 256   8  24   3   1   3]': '5a22ce47c71183713d89124ceeca115c3cfe446828efa80912aa2e9b51aa3b58',
	'[  1   1   6   4   4   1   1 512   8   6   4   1   5]': '18f199da03378b6420356f8f1b1fbcd8b55a4c51dd36132aeabc0c2ad7ec46dc',
	'[ 2 16  4  2  4  1  1 64  6  2  4  2  1]': '3e38c2b3485097cf65764b3e15c0fed2e11c92f8e123f3d090b3a03879b07cc7',
	'[  4   1   4   7   2   1   1 256   6  22   2   1   1]': '785cf0b1b0387415ce536e036247679877a4fbf04fd1c133467ab6e4026c1f86',
	'[  2  16   7   2   8   7   7 512   8  14   4   2   3]': 'c0f1f375ed80e2b4d5e7c3ddb285f3b88889a0b3833a3a99f160e3c68aa75919',
	'[2 1 4 4 8 7 7 1 1 6 4 1 3]': 'becb017f478dc36c7ce3f18c16c91220bc4f7ffcaae8c93e2c00f7b9143ffa53',
	'[ 1  1  5  8  1  1  1  1 10  4  3  1  4]': '8f8f65c4b2a99cb8157d6d3065aa6bac7da554c7f3a29e7bebfb95f0c211db50',
	'[  1   1   8   4   4   5   5 128  14  12   4   1   6]': '9299a344b832998c444ca230badef7d81913ad60bab04fa5bc6f15ceeced1de4',
	'[  4  16   6   5   8   3   3 256  14   1   3   3   1]': '5d7500e3c83aa721a98351a4e351a1e228ee2a02403762a1fd063057b4d451b4',
	'[  4  16   2   8   2   1   1 512   6   2   4   1   1]': '7e05f96ee965730d67864cf58cc6c5b70183b37e5cc850d91e142e7b44519c40',
	'[  4   1   2   1   4   7   7 256   1  24   3   1   3]': '94409f18bb65b95b0f59fb963cc35fd2967c19007d7ecfee5f55eeb250e98824',
	'[  4  16   4   1   8   3   3 512  20   2   1   2   4]': 'a73ce72f7e09f338f37cb60365132d4d67a484c8fd03d6c7f10156878066a749',
	'[ 2  1  3  4  2  3  3  1  4 16  1  1  1]': '6a300643f490b5003d3a8cb9e3fa8fff15ef6916829c4e2e9e11e3ab353f1804',
	'[  4   1   8   2   2   3   3 128   2   8   3   1   5]': 'e3b6b0dbaeb240c34f71332084f7730fae68a90ff84f2b6228a9630e80b5903a',
	'[ 4 16  2  5  1  3  3 64 16 10  3  1  4]': '7854ed6a051df9f7735a080270a2cb696ae4695ae91ace7b533cc2c95165a808',
	'[  1  16   5   1   1   7   7 512  24  12   1   3   1]': '5291e8ff56d64aaf50b6b253f11fb0b9b540d19d9a503aa5c8819ce1c1b34fff',
	'[  2   1   5   1   4   3   3 512  20  24   2   1   2]': 'cb08b155315caf71debb73a8402b7157ce9b15bac15bc2efd0a0509701129541'}

spring_embedding = np.array([2e0, 16e0, 8e0, 4e0, 8e0, 3e0, 3e0, 1e0, 12e0, 24e0, 4e0, 1e0, 1e0])

accel_dataset = pickle.load(open('../boshcode/accel_dataset/accel_dataset_mini.pkl', 'rb'))
# accel_dataset = {}

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


for cnn_name, accel_hash in manual_accel_hashes.items():
	cnn_hash = manual_models[cnn_name]
	flag = True
	if accel_hash + '.pkl' not in os.listdir(accel_models_dir): 
		print(f'Accelerator hash: {accel_hash} is not trained')
		flag = False
	if 'model.pt' not in os.listdir(os.path.join(cnn_models_dir, cnn_hash)): 
		print(f'CNN hash: {cnn_hash} is not trained')
		flag = False

	if accel_hash not in accel_dataset.keys() and flag:
		print(f'Adding accelerator hash: {accel_hash} to dataset')
		accel_dataset[accel_hash] = \
			{'cnn_hash': cnn_hash, 'accel_emb': spring_embedding, \
			'train_acc': None, 'val_acc': None, 'test_acc': None, 'latency': None, 'area': None, \
			'dynamic_energy': None, 'leakage_energy': None}

for accel_emb_str, accel_hash in mobilenet_accel_hashes.items():
	accel_emb = np.array([int(num) for num in accel_emb_str[1:-1].split()])
	cnn_hash = manual_models['mobilenet']
	flag = True
	if accel_hash + '.pkl' not in os.listdir(accel_models_dir): 
		print(f'Accelerator hash: {accel_hash} is not trained')
		flag = False
	if 'model.pt' not in os.listdir(os.path.join(cnn_models_dir, cnn_hash)): 
		print(f'CNN hash: {cnn_hash} is not trained')
		flag = False

	if accel_hash not in accel_dataset.keys() and flag:
		print(f'Adding accelerator hash: {accel_hash} to dataset')
		accel_dataset[accel_hash] = \
			{'cnn_hash': cnn_hash, 'accel_emb': spring_embedding, \
			'train_acc': None, 'val_acc': None, 'test_acc': None, 'latency': None, 'area': None, \
			'dynamic_energy': None, 'leakage_energy': None}

accel_dataset_file = '../boshcode/accel_dataset/accel_dataset_mini_bkp.pkl'
print(f'Saving dataset to: {accel_dataset_file}')
pickle.dump(accel_dataset, open(accel_dataset_file, 'wb+'), pickle.HIGHEST_PROTOCOL)

