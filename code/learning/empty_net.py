
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

class Empty_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Empty_Net, self).__init__()

		if learning_module is "DeepSet":
			self.model = DeepSet(
				param.rl_phi_network_architecture,
				param.rl_rho_network_architecture,
				param.rl_network_activation
				)

			self.action_dim = param.rl_rho_network_architecture[-1].out_features

	def policy(self,x):

		# inputs observation from all agents...
		# outputs policy for all agents

		n_agents = np.array(x).shape[0]
		x = np.squeeze(x)
		A = []
		for i in range(n_agents):
			x_i = [x[i]]
			A.append(self.model.forward(x_i).detach().numpy())
		A = np.reshape(np.asarray(A).flatten(),(n_agents,self.action_dim))
		return A

	def __call__(self,x):			
		return self.model(x)



