import os
import pandas as pd 
import numpy as np
import torch
import random
from constants import *
import matplotlib.pyplot as plt

def save_model(model, optimizer, epoch, loss_list, path):
	file_path = os.path.join(path, model.name + ".ckpt")
	if not os.path.exists(path): os.makedirs(path)
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_list': loss_list}, file_path)

def load_model(model, optimizer, path):
	file_path = os.path.join(path, model.name + ".ckpt")
	assert os.path.exists(file_path)
	checkpoint = torch.load(file_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss_list = checkpoint['loss_list']
	return model, optimizer, epoch, loss_list

def plotgraph(l, name, plotline=True):
	plt.plot(l, '.')
	if plotline: plt.plot(np.convolve(l, np.ones(5)/5, mode='same'), '--')
	plt.savefig(f'{name}.png')
	plt.cla()

def freeze_models(models):
	for model in models:
		for param in model.parameters(): param.requires_grad = False

def unfreeze_models(models):
	for model in models:
		for param in model.parameters(): param.requires_grad = True

def early_stop(tloss, vloss):
	if len(vloss) > 10:
		for i in range(1, 8):
			if vloss[-i] < vloss[-i-1]:
				return False
		return True
	return False