
# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import concurrent

# my package
from learning.deepset import DeepSet
from learning.feedforward import FeedForward

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
		
		self.phi_max = param.phi_max
		self.phi_min = param.phi_min
		# self.a_noise = param.a_noise

		self.psi = FeedForward(
			param.il_psi_network_architecture,
			param.il_network_activation)

		self.dim_g = param.il_psi_network_architecture[0].in_features - \
						self.model_obstacles.rho.out_dim - \
						self.model_neighbors.rho.out_dim
		self.device = torch.device('cpu')

	def to(self, device):
		self.device = device
		self.model_neighbors.to(device)
		self.model_obstacles.to(device)
		self.psi.to(device)
		return super().to(device)

	def policy(self,x,transformations):

		# inputs observation from all agents...
		# outputs policy for all agents

		A = np.empty((len(x),self.action_dim_per_agent))
		for i,x_i in enumerate(x):
			R = transformations[i][0]
			a_i = self(torch.Tensor(x_i))
			a_i = a_i.detach().numpy()
			a_i = np.matmul(R.T,a_i.T).T
			A[i,:] = a_i
		return A

	def export_to_onnx(self, filename):
		self.model_neighbors.export_to_onnx("{}_neighbors".format(filename))
		self.model_obstacles.export_to_onnx("{}_obstacles".format(filename))
		self.psi.export_to_onnx("{}_psi".format(filename))

	def __call__(self,x):
		# batches are grouped by number of neighbors (i.e., each batch has data with the same number of neighbors)
		# x is a 2D tensor, where the columns are: relative_goal, relative_neighbors, ...
		if self.dim_g == 2 and self.model_neighbors.phi.in_dim == 2:
			num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
			num_obstacles = int((x.size()[1] - 3 - 2*num_neighbors)/2)

			# print("neighbors ", num_neighbors)
			# print("obs ", num_obstacles)

			rho_neighbors = self.model_neighbors.forward(x[:,3:3+2*num_neighbors])
			
			# rho_neighbors = torch.zeros((len(x),16), device=self.device)

			# print("rho_neighbors", rho_neighbors)
			rho_obstacles = self.model_obstacles.forward(x[:,3+2*num_neighbors:])
			g = x[:,1:3]
		elif self.dim_g == 4 and self.model_neighbors.phi.in_dim == 4:
			num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
			num_obstacles = int((x.size()[1] - 5 - 4*num_neighbors)/2)

			# print("neighbors ", num_neighbors)
			# print("obs ", num_obstacles)

			rho_neighbors = self.model_neighbors.forward(x[:,5:5+4*num_neighbors])
			# print("rho_neighbors", rho_neighbors)
			rho_obstacles = self.model_obstacles.forward(x[:,5+4*num_neighbors:])
			g = x[:,1:5]
		elif self.dim_g == 2 and self.model_neighbors.phi.in_dim == 4:
			num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
			num_obstacles = int((x.size()[1] - 3 - 4*num_neighbors)/2)

			rho_neighbors = self.model_neighbors.forward(x[:,3:3+4*num_neighbors])
			# print("rho_neighbors", rho_neighbors)
			rho_obstacles = self.model_obstacles.forward(x[:,3+4*num_neighbors:])
			g = x[:,1:3]
		else:
			assert(False)
		# g_norm = g.norm(dim=1,keepdim=True)
		# time_to_goal = x[:,4:5]

		x = torch.cat((rho_neighbors, rho_obstacles, g),1)
		x = self.psi(x)

		# x = self.scale(x)

		return x

	def scale(self,x):
		x = torch.tanh(x) # x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((self.phi_max-self.phi_min)).float()+torch.tensor((self.phi_min)).float() #, x \in [amin,amax]
		return x
