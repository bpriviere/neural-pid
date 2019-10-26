
# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

# my package
from learning.deepset import DeepSet

class Barrier_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Barrier_Net, self).__init__()

		if learning_module is "DeepSet":
			self.model = DeepSet(
				param.rl_phi_network_architecture,
				param.rl_rho_network_architecture,
				param.rl_network_activation, 
				param.a_max,
				param.a_min)

			self.action_dim = param.rl_rho_network_architecture[-1].out_features
			self.state_dim = param.rl_phi_network_architecture[0].in_features
			self.r_agent = param.r_agent
			self.a_max = param.a_max
			self.a_min = param.a_min 

	def policy(self,x):

		# inputs observation from all agents
		# outputs policy for all agents

		A = np.empty((len(x),self.action_dim))
		for i,x_i in enumerate(x):
			a_i = self([x_i])
			A[i,:] = a_i.detach().numpy()
		return A

	def __call__(self,x):
		a_wbarrier = self.model(x)

		barrier = torch.zeros((len(x),self.action_dim))
		for k,x_i in enumerate(x): 
			n_n = len(x_i.relative_neighbors)
			for relative_neighbor in x_i.relative_neighbors:
				p_ji = relative_neighbor[0:2]
				barrier[k,:] += -1/n_n*torch.tensor(
					p_ji/np.power(np.linalg.norm(p_ji)-self.r_agent,1))

		a_wbarrier += 0.1*barrier

		# control lim
		# scaling_factor = torch.max(
		# 	torch.tensor(self.a_max) / torch.max(torch.abs(a_wbarrier)))
		# if scaling_factor < 1:
		# 	a_wbarrier = scaling_factor
		return a_wbarrier



