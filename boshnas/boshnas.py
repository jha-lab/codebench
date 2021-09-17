import random
import multiprocessing
from joblib import Parallel, delayed
from copy import deepcopy
import random

from models import *
from constants import *
from acq import *
from adahessian import Adahessian
from boshnas_utils import *

NUM_CORES = multiprocessing.cpu_count()
DEBUG = False

class BOSHNAS():
	def __init__(self, input_dim, bounds, trust_region, second_order, parallel, model_aleatoric, save_path, pretrained):
		assert bounds[0].shape[0] == input_dim and bounds[1].shape[0] == input_dim
		self.input_dim = input_dim
		self.bounds = (torch.tensor(bounds[0], dtype=torch.float), torch.tensor(bounds[1], dtype=torch.float))
		self.parallel = parallel
		self.trust_region = trust_region
		self.second_order = second_order
		self.run_aleatoric = model_aleatoric
		self.path = save_path
		self.init_models(pretrained)

	def init_models(self, pretrained):
		self.student = student(self.input_dim)
		self.teacher = teacher(self.input_dim)
		self.student_opt = torch.optim.SGD(self.student.parameters() , lr=2*LR)
		self.teacher_opt = torch.optim.AdamW(self.teacher.parameters() , lr=2*LR)
		self.student_l, self.teacher_l = [], []
		self.epoch = 0
		if self.run_aleatoric:
			self.npn = npn(self.input_dim)
			self.npn_opt = torch.optim.AdamW(self.npn.parameters() , lr=0.05*LR)
			self.npn_l = []
		if pretrained:
			self.student, self.student_opt, _, self.student_l = load_model(self.student, self.student_opt, self.path)
			self.teacher, self.teacher_opt, self.epoch, self.teacher_l = load_model(self.teacher, self.teacher_opt, self.path)
			if self.run_aleatoric:
				self.npn, self.npn_opt, self.epoch, self.npn_l = load_model(self.npn, self.npn_opt, self.path)

	def train(self, xtrain, ytrain):
		global EPOCHS
		self.epoch += EPOCHS
		teacher_loss = self.train_teacher(xtrain, ytrain)
		student_loss = self.train_student(xtrain, ytrain)
		npn_loss = self.train_npn(xtrain, ytrain) if self.run_aleatoric else 0
		if DEBUG:
			plotgraph(self.student_l, 'student'); plotgraph(self.teacher_l, 'teacher')
			if self.run_aleatoric: plotgraph(self.npn_l, 'npn')
		return npn_loss, teacher_loss, student_loss

	def predict(self, x):
		'''
		input: x - list of queries
		outputs: (predictions, uncertainties) where uncertainties is a pair of al and ep uncertainties
		'''
		if not self.run_aleatoric: self.teacher.eval()
		with torch.no_grad():
			outputs = []
			for feat in x:
				feat = torch.tensor(feat, dtype=torch.float)
				if self.run_aleatoric:
					pred, al = self.npn(feat)
				else:
					pred, al = self.teacher(feat), 0
				ep = self.student(feat)
				outputs.append((pred, (ep, al)))
		if not self.run_aleatoric: self.teacher.train()
		return outputs

	def get_queries(self, x, k, explore_type, use_al=False):
		'''
		x = list of inputs
		k = integer (batch of queries i.e. index of x closest my opt output)
		explore_type (str, optional): type of exploration; one
			in ['ucb', 'ei', 'pi', 'ts', 'percentile', 'mean', 
			'confidence', 'its', 'unc']. 'unc' added for purely
			uncertainty-based sampling 
		'''
		threads = max(NUM_CORES, k) if self.parallel else 1
		inits = random.choices(x, k=k)
		if not self.run_aleatoric: self.teacher.eval()
		self.freeze()
		inits = Parallel(n_jobs=threads, backend='threading')(delayed(self.parallelizedFunc)(ind, i, explore_type, use_al) \
			for ind, i in enumerate(inits))
		self.unfreeze();
		indices = []
		for init in inits:
			devs = torch.mean(torch.abs(init - torch.from_numpy(np.array(x))), dim=1)
			indices.append(torch.argmin(devs).item())
		if not self.run_aleatoric: self.teacher.train()
		return indices

	def parallelizedFunc(self, ind, init, explore_type, use_al):
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		optimizer = torch.optim.SGD([init] , lr=10) if not self.second_order else Adahessian([init] , lr=0.5)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
		iteration = 0; equal = 0; z_old = 100; z = 0; zs = []
		while iteration < 200:
			old = deepcopy(init.data)
			if self.trust_region:
				trust_bounds = (old*(1-trust_region), old*(1+trust_region))
			pred, al = self.npn(init) if self.run_aleatoric else (self.teacher(init), 0)
			ep = self.student(init)
			if self.run_aleatoric and use_al: ep = torch.add(ep, al)
			z = gosh_acq(pred, ep)
			zs.append(z.item())
			optimizer.zero_grad()
			z.backward(create_graph=True)
			optimizer.step()
			scheduler.step()
			init.data = torch.max(self.bounds[0], torch.min(self.bounds[1], init.data))
			if self.trust_region:
				init.data = torch.max(trust_bounds[0], torch.min(trust_bounds[1], init.data))
			equal = equal + 1 if abs(z.item() - z_old) < epsilon else 0
			if equal > 5: break
			iteration += 1; z_old = deepcopy(z.item())
		if DEBUG:
			assert self.parallel == False
			plotgraph(zs, f'aqn_scores_{ind}', plotline=False)
		init.requires_grad = False 
		return init.data

	def freeze(self):
		freeze_models([self.student, self.teacher])
		if self.run_aleatoric: freeze_models([self.npn])

	def unfreeze(self):
		unfreeze_models([self.student, self.teacher])
		if self.run_aleatoric: unfreeze_models([self.npn])

	## Teacher training
	def train_teacher_helper(self, tset, training = True):
		total = 0
		for feat, y_true in tset:
			feat = torch.tensor(feat, dtype=torch.float)
			y_true = torch.tensor([y_true], dtype=torch.float)
			y_pred = self.teacher(feat)
			loss = (y_pred - y_true) ** 2
			if training:
				self.teacher_opt.zero_grad(); loss.backward(); self.teacher_opt.step()
			total += loss
		return total.item() / len(tset)

	def train_teacher(self, xtrain, ytrain):
		dset = list(zip(xtrain, ytrain))
		vloss = []; 
		for _ in range(EPOCHS):
			random.shuffle(dset); 
			split = int(Train_test_split * len(dset))
			tset, vset = dset[:split], dset[split:]
			self.teacher_l.append(self.train_teacher_helper(tset))
			vloss.append(self.train_teacher_helper(vset, False))
			if early_stop(self.teacher_l, vloss): break
		save_model(self.teacher, self.teacher_opt, self.epoch, self.teacher_l, self.path)
		return self.teacher_l[-1]

	## Student training
	def train_student_helper(self, tset, training = True):
		total = 0
		for feat, y_true in tset:
			feat = torch.tensor(feat, dtype=torch.float)
			outputs = [self.teacher(feat) for _ in range(Teacher_student_cycles)]
			y_true = torch.std(torch.stack(outputs))
			y_pred = self.student(feat)
			loss = (y_pred - y_true) ** 2
			if training:
				self.student_opt.zero_grad(); loss.backward(); self.student_opt.step()
			total += loss
		return total.item() / len(tset)

	def train_student(self, xtrain, ytrain):
		dset = list(zip(xtrain, ytrain))
		vloss = []; 
		for _ in range(EPOCHS):
			random.shuffle(dset); 
			split = int(Train_test_split * len(dset))
			tset, vset = dset[:split], dset[split:]
			self.student_l.append(self.train_student_helper(tset))
			vloss.append(self.train_student_helper(vset, False))
			if early_stop(self.student_l, vloss): break
		save_model(self.student, self.student_opt, self.epoch, self.student_l, self.path)
		return self.student_l[-1]

	## NPN training
	def train_npn_helper(self, tset, training = True):
		total = 0
		for feat, y_true in tset:
			feat = torch.tensor(feat, dtype=torch.float)
			y_pred = self.npn(feat)
			loss = Aleatoric_Loss(y_pred, y_true)
			if training:
				self.npn_opt.zero_grad(); loss.backward(); self.npn_opt.step()
			total += loss
		return total.item() / len(tset)

	def train_npn(self, xtrain, ytrain):
		dset = list(zip(xtrain, ytrain))
		vloss = []; 
		for _ in range(EPOCHS):
			random.shuffle(dset); 
			split = int(Train_test_split * len(dset))
			tset, vset = dset[:split], dset[split:]
			self.npn_l.append(self.train_npn_helper(tset))
			vloss.append(self.train_npn_helper(vset, False))
			if early_stop(self.npn_l, vloss): break
		save_model(self.npn, self.npn_opt, self.epoch, self.npn_l, self.path)
		return self.npn_l[-1]