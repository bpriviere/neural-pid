
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
			self.model_neighbors = DeepSet(
				param.il_phi_network_architecture,
				param.il_rho_network_architecture,
				param.il_network_activation,
				param.env_name
				)
			self.model_obstacles = DeepSet(
				param.il_phi_obs_network_architecture,
				param.il_rho_obs_network_architecture,
				param.il_network_activation,
				param.env_name
				)

			self.action_dim_per_agent = param.il_psi_network_architecture[-1].out_features
		
		self.a_max = param.a_max
		self.a_min = param.a_min
		self.a_noise = param.a_noise

		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation

	def policy(self,x,transformations):

		# inputs observation from all agents...
		# outputs policy for all agents
		A = np.empty((len(x),self.action_dim_per_agent))
		for i,x_i in enumerate(x):
			
			
			# R = transformations[i][0]
			# print('R: ', R)
			# print('transformations: ', transformations)
			# exit()

			R = transformations[i][0]

			a_i = self(torch.Tensor(x_i))
			a_i = a_i.detach().numpy()
			a_i = np.matmul(R.T,a_i.T).T
			
			# a_i = self(torch.Tensor([x_i]))
			# a_i = a_i + self.a_noise*np.random(size=a_i.shape)
			
			A[i,:] = a_i
		return A

	def __call__(self,x):
		# batches are grouped by number of neighbors (i.e., each batch has data with the same number of neighbors)
		# x is a 2D tensor, where the columns are: relative_goal, relative_neighbors, ...

		num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
		num_obstacles = int((x.size()[1] - 5 - 4*num_neighbors)/2)

		idx_neighbors = np.arange(5,5+4*num_neighbors,dtype=int)
		idx_obstacles = np.arange(5+4*num_neighbors,x.shape[1],dtype=int)
		idx_goal = np.arange(1,3,dtype=int)

		# print('x[0,:]: ', x[0,:])
		# print('x[0,5:5+4*num_neighbors]: ', x[0,5:5+4*num_neighbors])
		# print('x[0,idx_neighbors]: ', x[0,idx_neighbors])
		# print('x[0,5+4*num_neighbors:]: ', x[0,5+4*num_neighbors:])
		# print('x[0,idx_obstacles]: ', x[0,idx_obstacles])
		# print('x[0,1:3]:', x[0,1:3])
		# print('x[0,idx_goal]: ', x[0,idx_goal])
		# exit()

		# print("neighbors ", num_neighbors)
		# print("obs ", num_obstacles)

		rho_neighbors = self.model_neighbors.forward(x[:,5:5+4*num_neighbors])
		# print("rho_neighbors", rho_neighbors)
		rho_obstacles = self.model_obstacles.forward(x[:,5+4*num_neighbors:])
		g = x[:,1:3]
		# g_norm = g.norm(dim=1,keepdim=True)
		# time_to_goal = x[:,4:5]
		x = torch.cat((rho_neighbors, rho_obstacles, g),1)

		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)

		# add noise ???
		# x = x + self.a_noise*np.random(size=x.shape)

		# if no control authority lim in deepset implement here instead:
		# x = torch.tanh(x) # x \in [-1,1]
		# x = (x+1.)/2.*torch.tensor((self.a_max-self.a_min)).float()+torch.tensor((self.a_min)).float() #, x \in [amin,amax]
		return x
