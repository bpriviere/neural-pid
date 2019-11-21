
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np 

class DeepSet(nn.Module):

	def __init__(self,phi_layers,rho_layers,activation,env_name):
		super(DeepSet, self).__init__()
		
		self.phi = Phi(phi_layers,activation)
		self.rho = Rho(rho_layers,activation)

		self.state_dim = phi_layers[0].in_features
		self.phi_out_dim = phi_layers[-1].out_features
		self.rho_in_dim = rho_layers[0].in_features
		self.action_dim = rho_layers[-1].out_features

		self.env_name = env_name

	def forward(self,x):

		if self.env_name == 'Consensus':
			return self.consensus_forward(x)
		elif self.env_name == 'SingleIntegrator':
			return self.si_forward(x)

	def consensus_forward(self,x):
		# x is a list of namedtuple with <relative_neighbors> where relative_neighbors is a list
		X = torch.zeros((len(x),self.hidden_dim))
		for step,x_i in enumerate(x):
			relative_neighbors = x_i
			summ = torch.zeros((self.hidden_dim))
			for relative_neighbor in relative_neighbors:
				relative_neighbor = np.array(relative_neighbor, ndmin=1)
				relative_neighbor = torch.from_numpy(relative_neighbor).float()
				summ += self.phi(relative_neighbor)
			X[step,:] = summ
		out = self.rho(X)
		return out

	def si_forward(self,x):
		# x is a list of namedtuple with <relative_goal, relative_neighbors> where relative_neighbors is a list
		X = torch.zeros((len(x),self.rho_in_dim))
		for step,x_i in enumerate(x):
			relative_goal = torch.from_numpy(x_i.relative_goal).float() 
			relative_neighbors = x_i.relative_neighbors
			summ = torch.zeros((self.phi_out_dim))
			for relative_neighbor in relative_neighbors:
				relative_neighbor = torch.from_numpy(relative_neighbor).float()
				summ += self.phi(relative_neighbor)
			X[step,:] = torch.cat((relative_goal,summ))
		return self.rho(X)

class Phi(nn.Module):

	def __init__(self,layers,activation):
		super(Phi, self).__init__()
		self.layers = layers
		self.activation = activation
	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x


class Rho(nn.Module):
	def __init__(self,layers,activation):
		super(Rho, self).__init__()
		self.layers = layers
		self.activation = activation

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x

