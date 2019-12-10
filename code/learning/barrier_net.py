
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
from learning.empty_net import Empty_Net

class Barrier_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Barrier_Net, self).__init__()

		self.model = DeepSet(
			param.il_phi_network_architecture,
			param.il_rho_network_architecture,
			param.il_network_activation,
			param.env_name
			)

		self.r_agent = param.r_agent
		self.action_dim_per_agent = param.il_rho_network_architecture[-1].out_features
		self.state_dim_per_agent = param.il_phi_network_architecture[0].in_features
		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation
		self.a_max = param.a_max
		self.a_min = param.a_min 
		self.Ds = 2*param.r_agent
		self.b_gamma = param.b_gamma 

	def policy(self,x):

		# inputs observation from all agents...
		# outputs policy for all agents

		A = np.empty((len(x),self.action_dim_per_agent))
		for i,x_i in enumerate(x):
			a_i = self(torch.Tensor([x_i]))
			A[i,:] = a_i.detach().numpy()
		return A

	def empty(self,x):
		rho = self.model.forward(x)
		g = x[:,0:2]
		x = torch.cat((rho, g),1)
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x

	def __call__(self,x):

		empty_action = self.empty(x)
		barrier_action = torch.zeros((len(x),self.action_dim_per_agent))

		for i,observation_i in enumerate(x):
			n_n = int(len(observation_i)/(self.state_dim_per_agent)-1)

			for j in range(n_n):
				# j+1 to skip relative goal entries 
				idx = self.state_dim_per_agent*(j+1)+np.arange(0,self.state_dim_per_agent,dtype=int)
				relative_neighbor = observation_i[idx]
				p_ij = -1*relative_neighbor[0:2]
				v_ij = -1*relative_neighbor[2:]
				barrier_action[i,:] += self.get_b_ij(p_ij,v_ij)

		# scale actions 
		action = empty_action + barrier_action 
		action = torch.tanh(action) # action \in [-1,1]
		action = (action+1.)/2.*torch.tensor((self.a_max-self.a_min)).float()+torch.tensor((self.a_min)).float() # action \in [amin,amax]

		return action 

	def get_B_i():
		pass


	def get_b_ij(self,dp,dv):
		
		"""
		dp = p^i - p^j
		dv = v^i - v^j
		"""
		h_ij = self.get_h_ij(dp,dv)
		# return self.b_gamma/(np.power(h_ij,3))
		return 1/h_ij


	def get_h_ij(self,dp,dv):
		h_ij = np.sqrt(4*self.a_max*(np.linalg.norm(dp) - self.Ds)) \
			+ np.matmul(dv.T, dp)/np.linalg.norm(dp)
		return h_ij