
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np 

class DeepSet(nn.Module):

	def __init__(self, network_architecture_phi, network_architecture_rho, activation,a_max,a_min):
		super(DeepSet, self).__init__()
		
		self.phi = Phi(network_architecture_phi,activation)
		self.rho = Rho(network_architecture_rho,activation,a_max,a_min)

		self.state_dim = network_architecture_phi[0].in_features
		self.hidden_dim = network_architecture_phi[-1].out_features
		self.action_dim = network_architecture_rho[-1].out_features

	def forward(self,x):
		# x is a list of namedtuple with <relative_goal, relative_neighbors> where relative_neighbors is a list
		X = torch.zeros((len(x),self.hidden_dim+self.state_dim))
		for step,x_i in enumerate(x):
			relative_goal = torch.from_numpy(x_i.relative_goal).float() 
			relative_neighbors = x_i.relative_neighbors
			summ = torch.zeros((self.hidden_dim))
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

	def __init__(self,layers,activation,a_max,a_min):
		super(Rho, self).__init__()
		self.layers = layers
		self.activation = activation
		self.a_max = a_max
		self.a_min = a_min 

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = torch.tanh(self.layers[-1](x)) #, x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((self.a_max-self.a_min)).float()+torch.tensor((self.a_min)).float() #, x \in [amin,amax]
		return x

