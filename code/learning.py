
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from param import param 
from numpy import squeeze, array

class PPO(nn.Module):
	def __init__(self):
		super(PPO, self).__init__()
		self.data = []
		
		self.fc1   = nn.Linear(param.get('sys_n'),256)
		self.fc_pi = nn.Linear(256,param.get('sys_card_A'))
		self.fc_v  = nn.Linear(256,1)
		self.optimizer = optim.Adam(self.parameters(), lr=param.get('rl_lr'))
		self.actions = param.get('sys_actions')		

	def pi(self, x, softmax_dim = 0):
		x = F.relu(self.fc1(x))
		x = self.fc_pi(x)
		prob = F.softmax(x, dim=softmax_dim)
		return prob
	
	def v(self, x):
		x = F.relu(self.fc1(x))
		v = self.fc_v(x)
		return v
	  
	def put_data(self, transition):
		self.data.append(transition)
		
	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
		for transition in self.data:
			s, a, r, s_prime, prob_a, done = transition
			
			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			prob_a_lst.append([prob_a])
			done_mask = 0 if done else 1
			done_lst.append([done_mask])
			
		s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
										  torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
										  torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
		self.data = []
		return s, a, r, s_prime, done_mask, prob_a
		
	def train_net(self):
		s, a, r, s_prime, done_mask, prob_a = self.make_batch()

		for i in range(param.get('rl_K_epoch')):
			td_target = r + param.get('rl_gamma') * self.v(s_prime) * done_mask
			delta = td_target - self.v(s)
			delta = delta.detach().numpy()

			advantage_lst = []
			advantage = 0.0
			for delta_t in delta[::-1]:
				advantage = param.get('rl_gamma') * param.get('rl_lmbda') * advantage + delta_t[0]
				advantage_lst.append([advantage])
			advantage_lst.reverse()
			advantage = torch.tensor(advantage_lst, dtype=torch.float)

			pi = self.pi(s, softmax_dim=1)
			pi_a = pi.gather(1,a)
			ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1-param.get('rl_eps_clip'), 1+param.get('rl_eps_clip')) * advantage
			loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

	def policy(self, state):
		prob = self.pi(torch.from_numpy(state).float())
		m = Categorical(prob)
		classification = m.sample().item()
		return self.class_to_force(classification)

	def class_to_force(self, a):
		return self.actions[a]

class GainsNet(nn.Module):
	"""
	neural net to predict gains, kp, kd, from state, s
	"""
	def __init__(self):
		super(GainsNet, self).__init__()
		self.fc1 = nn.Linear(param.get('sys_n'), 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 3)

	def forward(self, x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		state = x
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = (x[:,0]*state[:,0] + x[:,0]*state[:,1] + \
			x[:,1]*state[:,2] + x[:,1]*state[:,3]) 
		x = x.reshape((len(x),1))
		return x

	def policy(self,state):
		action = self(torch.from_numpy(state).float())
		action = squeeze(action.detach().numpy())
		return action

	def get_kp(self,x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = x[:,0].detach().numpy()
		return x

	def get_kd(self,x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = x[:,1].detach().numpy()
		return x

class PIDNet(nn.Module):
	"""
	neural net to predict gains, kp, kd, from state, s
	"""
	def __init__(self):
		super(PIDNet, self).__init__()
		self.fc1 = nn.Linear(param.get('sys_n'), 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, param.get('sys_n') + 2)

	def forward(self, x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		state = x
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		ref_state = x[:,2:]
		error = state-ref_state
		x = x[:,0]*error[:,0] + x[:,0]*error[:,1] + \
			x[:,1]*error[:,2] + x[:,1]*error[:,3] 
		x = x.reshape((len(x),1))
		return x

	def policy(self,state):
		action = self(torch.from_numpy(state).float())
		action = squeeze(action.detach().numpy())
		return action

	def get_kp(self,x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = x[:,0].detach().numpy()
		return x

	def get_kd(self,x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = x[:,1].detach().numpy()
		return x

	def get_ref_state(self,x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = x[:,1:].detach().numpy()
		return x