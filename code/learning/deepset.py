
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np 

class DeepSet(nn.Module):

	def __init__(self, network_architecture_phi, network_architecture_rho, activation):
		super(DeepSet, self).__init__()
		
		self.phi = Phi(network_architecture_phi,activation)
		self.rho = Rho(network_architecture_rho,activation)

		self.state_dim = network_architecture_phi[0].in_features
		self.hidden_dim = network_architecture_phi[-1].out_features
		self.action_dim = network_architecture_rho[-1].out_features

	def forward(self,x):
		# x is a list of [[s^i,{s^j}]] for all j neighbors

		X = torch.zeros((len(x),self.hidden_dim+2*self.state_dim))
		for step,x_i in enumerate(x):
			s_i,s_g,s_js = self.make_list(x_i)
			summ = torch.zeros((self.hidden_dim))
			for s_j in s_js:
				summ += self.phi(s_j)

			s_i = torch.from_numpy(s_i).float()
			s_g = torch.from_numpy(s_g).float()
			X[step,:] = torch.cat((s_i,s_g,summ))
		return self.rho(X)

	def make_list(self,x):
		# x is an array [s^i, {s^j}] for all j neighbors
		n_n = int(len(x)/self.state_dim)-2
		s_i = x[0:self.state_dim]
		s_g = x[self.state_dim:self.state_dim*2]
		s_js = []
		for i_n in range(n_n):
			idxs = (i_n+2)*self.state_dim + np.arange(0,self.state_dim)
			s_js.append(x[idxs])
		return s_i,s_g,s_js

class Phi(nn.Module):

	def __init__(self,layers,activation):
		super(Phi, self).__init__()
		self.layers = layers
		self.activation = activation

	def forward(self, x):

		if isinstance(x, (np.ndarray, np.generic) ):
			x = torch.from_numpy(x).float()

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

		if isinstance(x, (np.ndarray, np.generic) ):
			x = torch.from_numpy(x).float()

		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x