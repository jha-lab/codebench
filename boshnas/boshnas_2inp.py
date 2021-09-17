import random
import multiprocessing
from joblib import Parallel, delayed
from copy import deepcopy
import random

from models_2inp import *
from constants import *
from acq import *
from adahessian import Adahessian
from boshnas_utils import *

NUM_CORES = multiprocessing.cpu_count()
DEBUG = False

class BOSHNAS():
	def __init__(self, input_dim1, input_dim2, bounds1, bounds2, trust_region, second_order, parallel, \
		model_aleatoric, save_path, pretrained):
		assert bounds1[0].shape[0] == input_dim1 and bounds1[1].shape[0] == input_dim1
		assert bounds2[0].shape[0] == input_dim2 and bounds2[1].shape[0] == input_dim2
		self.input_dim1 = input_dim1
		self.input_dim2 = input_dim2
		self.bounds1 = (torch.tensor(bounds1[0], dtype=torch.float), torch.tensor(bounds1[1], dtype=torch.float))
		self.bounds2 = (torch.tensor(bounds2[0], dtype=torch.float), torch.tensor(bounds2[1], dtype=torch.float))
		self.parallel = parallel
		self.trust_region = trust_region
		self.second_order = second_order
		self.run_aleatoric = model_aleatoric
		self.path = save_path
		self.init_models(pretrained)

	def init_models(self, pretrained):
		self.student = student(self.input_dim1, self.input_dim2)
		self.teacher = teacher(self.input_dim1, self.input_dim2)
		self.student_opt = torch.optim.SGD(self.student.parameters() , lr=2*LR)
		self.teacher_opt = torch.optim.AdamW(self.teacher.parameters() , lr=2*LR)
		self.student_l, self.teacher_l = [], []
		self.epoch = 0
		if self.run_aleatoric:
			self.npn = npn(self.input_dim1, self.input_dim2)
			self.npn_opt = torch.optim.AdamW(self.npn.parameters() , lr=0.05*LR)
			self.npn_l = []
		if pretrained:
			self.student, self.student_opt, _, self.student_l = load_model(self.student, self.student_opt, self.path)
			self.teacher, self.teacher_opt, self.epoch, self.teacher_l = load_model(self.teacher, self.teacher_opt, self.path)
			if self.run_aleatoric:
				self.npn, self.npn_opt, self.epoch, self.npn_l = load_model(self.npn, self.npn_opt, self.path)

	def train(self, xtrain1, xtrain2, ytrain):
		global EPOCHS
		self.epoch += EPOCHS
		teacher_loss = self.train_teacher(xtrain1, xtrain2, ytrain)
		student_loss = self.train_student(xtrain1, xtrain2, ytrain)
		npn_loss = self.train_npn(xtrain1, xtrain2, ytrain) if self.run_aleatoric else 0
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
				feat1, feat2 = torch.tensor(feat[0], dtype=torch.float), torch.tensor(feat[1], dtype=torch.float)
				if self.run_aleatoric:
					pred, al = self.npn(feat1, feat2)
				else:
					pred, al = self.teacher(feat1, feat2), 0
				ep = self.student(feat1, feat2)
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
			devs1 = torch.mean(torch.abs(init[0] - torch.from_numpy(np.array([x[0] for x in x]))), dim=1)
			devs2 = torch.mean(torch.abs(init[1] - torch.from_numpy(np.array([x[1] for x in x]))), dim=1)
			indices.append((torch.argmin(devs1).item(), torch.argmin(devs2).item()))
		if not self.run_aleatoric: self.teacher.train()
		return indices

	def parallelizedFunc(self, ind, init, explore_type, use_al):
		init1 = torch.tensor(init[0], dtype=torch.float, requires_grad=True)
		init2 = torch.tensor(init[1], dtype=torch.float, requires_grad=True)
		optimizer = torch.optim.SGD([init1, init2] , lr=10) if not self.second_order else Adahessian([init1, init2] , lr=0.5)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
		iteration = 0; equal = 0; z_old = 100; z = 0; zs = []
		while iteration < 200:
			old1, old2 = deepcopy(init1.data), deepcopy(init2.data)
			if self.trust_region:
				trust_bounds1 = (old1*(1-trust_region), old1*(1+trust_region))
				trust_bounds2 = (old2*(1-trust_region), old2*(1+trust_region))
			pred, al = self.npn(init1, init2) if self.run_aleatoric else (self.teacher(init1, init2), 0)
			ep = self.student(init1, init2)
			if self.run_aleatoric and use_al: ep = torch.add(ep, al)
			z = gosh_acq(pred, ep)
			zs.append(z.item())
			optimizer.zero_grad()
			z.backward(create_graph=True)
			optimizer.step()
			scheduler.step()
			init1.data = torch.max(self.bounds1[0], torch.min(self.bounds1[1], init1.data))
			init2.data = torch.max(self.bounds2[0], torch.min(self.bounds2[1], init2.data))
			if self.trust_region:
				init1.data = torch.max(trust_bounds1[0], torch.min(trust_bounds1[1], init1.data))
				init2.data = torch.max(trust_bounds2[0], torch.min(trust_bounds2[1], init2.data))
			equal = equal + 1 if abs(z.item() - z_old) < epsilon else 0
			if equal > 5: break
			iteration += 1; z_old = deepcopy(z.item())
		if DEBUG:
			assert self.parallel == False
			plotgraph(zs, f'aqn_scores_{ind}', plotline=False)
		init1.requires_grad = False 
		init2.requires_grad = False 
		return (init1.data, init2.data)

	def freeze(self):
		freeze_models([self.student, self.teacher])
		if self.run_aleatoric: freeze_models([self.npn])

	def unfreeze(self):
		unfreeze_models([self.student, self.teacher])
		if self.run_aleatoric: unfreeze_models([self.npn])

	## Teacher training
	def train_teacher_helper(self, tset, training = True):
		total = 0
		for feat1, feat2, y_true in tset:
			feat1 = torch.tensor(feat1, dtype=torch.float)
			feat2 = torch.tensor(feat2, dtype=torch.float)
			y_true = torch.tensor([y_true], dtype=torch.float)
			y_pred = self.teacher(feat1, feat2)
			loss = (y_pred - y_true) ** 2
			if training:
				self.teacher_opt.zero_grad(); loss.backward(); self.teacher_opt.step()
			total += loss
		return total.item() / len(tset)

	def train_teacher(self, xtrain1, xtrain2, ytrain):
		dset = list(zip(xtrain1, xtrain2, ytrain))
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
		for feat1, feat2, y_true in tset:
			feat1 = torch.tensor(feat1, dtype=torch.float)
			feat2 = torch.tensor(feat2, dtype=torch.float)
			outputs = [self.teacher(feat1, feat2) for _ in range(Teacher_student_cycles)]
			y_true = torch.std(torch.stack(outputs))
			y_pred = self.student(feat1, feat2)
			loss = (y_pred - y_true) ** 2
			if training:
				self.student_opt.zero_grad(); loss.backward(); self.student_opt.step()
			total += loss
		return total.item() / len(tset)

	def train_student(self, xtrain1, xtrain2, ytrain):
		dset = list(zip(xtrain1, xtrain2, ytrain))
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
		for feat1, feat2, y_true in tset:
			feat1 = torch.tensor(feat1, dtype=torch.float)
			feat2 = torch.tensor(feat2, dtype=torch.float)
			y_pred = self.npn(feat1, feat2)
			loss = Aleatoric_Loss(y_pred, y_true)
			if training:
				self.npn_opt.zero_grad(); loss.backward(); self.npn_opt.step()
			total += loss
		return total.item() / len(tset)

	def train_npn(self, xtrain1, xtrain2, ytrain):
		dset = list(zip(xtrain1, xtrain2, ytrain))
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