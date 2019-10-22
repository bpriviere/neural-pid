
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
				param.rl_network_activation
				)

			self.action_dim = param.rl_rho_network_architecture[-1].out_features
			self.state_dim = param.rl_phi_network_architecture[0].in_features
			self.R = param.r_agent

	def policy(self,x):

		# inputs observation from all agents
		# outputs policy for all agents

		n_agents = np.array(x).shape[0]
		x = np.squeeze(x)
		A = []
		for i in range(n_agents):
			x_i = [x[i]]
			A.append(self(x_i).detach().numpy())
		A = np.reshape(np.asarray(A).flatten(),(n_agents,self.action_dim))
		return A

	def __call__(self,x):
		a_nom = self.model(x)
		a_wbarrier = a_nom 

		for k,x_i in enumerate(x): 
			n_n = int(len(x_i)/self.state_dim)-2 
			barrier = torch.zeros((self.action_dim)) 
			for i_n in range(n_n):
				idxs = (i_n+2)*self.state_dim + np.arange(0,self.state_dim/2)
				idxs = idxs.astype(int)
				barrier += torch.tensor(
					-x_i[idxs]/np.power(np.linalg.norm(x_i[idxs]) - self.R,3))
			a_wbarrier[k,:] += 0*barrier
		return a_wbarrier



