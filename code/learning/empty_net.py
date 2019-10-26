
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
				param.rl_network_activation,
				param.a_max,
				param.a_min
				)

			self.action_dim = param.rl_rho_network_architecture[-1].out_features

	def policy(self,x):

		# inputs observation from all agents...
		# outputs policy for all agents

		A = np.empty((len(x),self.action_dim))
		for i,x_i in enumerate(x):
			a_i = self([x_i])
			A[i,:] = a_i.detach().numpy()
		return A

	def __call__(self,x):			
		# if no control authority lim in deepset implement here instead:
		x = torch.tanh(self.model(x)) #, x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((self.a_max-self.a_min)).float()+torch.tensor((self.a_min)).float() #, x \in [amin,amax]
		return x



