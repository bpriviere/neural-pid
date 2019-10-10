

import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# adapted from 
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py

class DDPG():

	def __init__(self,
		state_dim,
		action_dim,
		control_lim,
		lr_mu,
		lr_q,
		gamma,
		batch_size,
		buffer_limit,
		action_std,
		tau,
		K_epoch,
		max_action_perturb,
		gpu_on):

		if gpu_on:
			self.gpu_on = True
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			print('Cuda Available: ', torch.cuda.is_available())
			torch.set_default_tensor_type('torch.cuda.FloatTensor')
		else:
			self.gpu_on = False
			self.device = "cpu"

		# some param
		self.action_dim = action_dim
		self.action_std = action_std
		self.control_lim = control_lim

		# hyperparameters
		self.lr_mu = lr_mu
		self.lr_q = lr_q
		self.gamma = gamma
		self.batch_size = batch_size
		self.buffer_limit = buffer_limit
		self.tau = tau
		self.K_epoch = K_epoch
		self.eps = max_action_perturb

		# memory
		self.data = ReplayBuffer(int(self.buffer_limit))
		# self.data = []

		# network
		self.q = QNet(state_dim,action_dim,self.device)
		self.q_target = QNet(state_dim,action_dim,self.device)
		self.q_target.load_state_dict(self.q.state_dict())		
		self.mu = MuNet(state_dim,action_dim,control_lim,self.device)
		self.mu_target = MuNet(state_dim,action_dim,control_lim,self.device)
		self.mu_target.load_state_dict(self.mu.state_dict())

		self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.lr_mu)
		self.q_optimizer  = optim.Adam(self.q.parameters(), lr=self.lr_q)


	# def make_batch(self):
	# 	s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
	# 	for transition in self.data:
	# 		s, a, r, s_prime, done = transition
			
	# 		s_lst.append(s)
	# 		a_lst.append([a])
	# 		r_lst.append([r])
	# 		s_prime_lst.append(s_prime)
	# 		done_mask = 0 if done else 1
	# 		done_lst.append([done_mask])
			
	# 	s = torch.tensor(s_lst, dtype=torch.float)
	# 	a = torch.tensor(a_lst, dtype=torch.float)
	# 	r = torch.tensor(r_lst, dtype=torch.float)
	# 	s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
	# 	done_mask = torch.tensor(done_lst, dtype=torch.float)
		
	# 	self.data = []
	# 	return s, a, r, s_prime, done_mask

	def train_net(self):
		s,a,r,s_prime,done_mask  = self.data.sample(self.batch_size)
		done_mask = done_mask.float()
		# s,a,r,s_prime,done_mask  = self.make_batch()
		
		for _ in range(self.K_epoch):
			target = r + self.gamma*(1-done_mask)*self.q_target(s_prime, self.mu_target(s_prime))

			q_loss = F.smooth_l1_loss(self.q(s,a), target.detach())
			self.q_optimizer.zero_grad()
			q_loss.backward()
			self.q_optimizer.step()
			
			mu_loss = -self.q(s,self.mu(s)).mean() # That's all for the policy loss.
			self.mu_optimizer.zero_grad()
			mu_loss.backward()
			self.mu_optimizer.step()

			# update target networks
			self.soft_update(self.mu,self.mu_target)
			self.soft_update(self.q,self.q_target)


	def soft_update(self,net,net_target):
		for param_target, param in zip(net_target.parameters(), net.parameters()):
			param_target.data.copy_(\
				self.tau*param_target.data+\
				(1.0-self.tau)*param.data*self.tau)

	
	def train_policy(self,s):
		# input: 
		# s, nd array, (n,)
		# output: 
		# a, nd array, (m,1)

		s = torch.from_numpy(s).float().to(self.device)
		a = self.mu(s).detach()
		b = torch.randn(self.action_dim)*self.action_std 
		b = torch.clamp(b,-self.eps,self.eps)
		a = a + b
		a = a.cpu().numpy()
		return a


	def policy(self,s):
		# input: 
		# s, nd array, (n,)
		# output: 
		# a, nd array, (m,1)
		
		s = torch.from_numpy(s).float().to(self.device)
		a = self.mu(s).detach()
		a = a.cpu().numpy()
		a = np.reshape(np.squeeze(a),(self.action_dim,1))

		return a


	def put_data(self,transition):
		# self.data.append(transition)
		self.data.put_data(transition)

# helper classes
class ReplayBuffer():
	def __init__(self,buffer_limit):
		self.buffer = collections.deque(maxlen=buffer_limit)
		self.count = 0

	def put_data(self, transition):
		self.buffer.append(transition)
		self.count += 1
	
	def sample(self, n):
		mini_batch = random.sample(self.buffer, n)
		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

		for transition in mini_batch:
			s, a, r, s_prime, done_mask = transition
			s_lst.append(s)
			a_lst.append(a)
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			done_mask_lst.append([done_mask])

		self.count = 0
		
		return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
			   torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
			   torch.tensor(done_mask_lst)
	
	def __len__(self):
		return self.count

# mu(s) = a
class MuNet(nn.Module):
	def __init__(self,state_dim,action_dim,control_lim,device):
		super(MuNet, self).__init__()

		self.to(device)

		self.control_lim = control_lim
		self.fc1 = nn.Linear(state_dim, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc_mu = nn.Linear(64, action_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mu = torch.tanh(self.fc_mu(x))*self.control_lim 
		return mu

# q = q(s,a)
class QNet(nn.Module):
	def __init__(self,state_dim,action_dim,device):
		super(QNet, self).__init__()
		
		self.to(device)

		self.fc_s = nn.Linear(state_dim, 32)
		self.fc_a = nn.Linear(action_dim,32)
		self.fc_q = nn.Linear(64, 32)
		self.fc_3 = nn.Linear(32,1)

	def forward(self, x, a):
		h1 = F.relu(self.fc_s(x))
		h2 = F.relu(self.fc_a(a))
		cat = torch.cat([h1,h2], dim=1)
		q = F.relu(self.fc_q(cat))
		q = self.fc_3(q)
		return q
