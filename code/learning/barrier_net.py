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
from utilities import torch_tile, min_dist_circle_rectangle, torch_min_point_circle_rectangle


class Barrier_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Barrier_Net, self).__init__()
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

		self.param = param 
		self.action_dim_per_agent = param.il_psi_network_architecture[-1].out_features
		self.state_dim_per_agent = param.il_phi_network_architecture[0].in_features
		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation
		self.device = torch.device('cpu')

	def to(self, device):
		self.device = device
		self.model_neighbors.to(device)
		self.model_obstacles.to(device)
		return super().to(device)


	def get_relative_positions_and_safety_functions(self,x):
		nd = x.shape[0] # number of data points in batch 
		nn = int(x[0,0].item()) # number of neighbors
		no = int((x.shape[1] - 1 - (nn+1)*self.state_dim_per_agent) / 2)  # number of obstacles 

		P = torch.zeros((nd,nn+no,2),device=self.device) # pj - pi 
		H = torch.zeros((nd,nn+no),device=self.device) 
		curr_idx = 0

		# print('new')
		for j in range(nn):
			# j+1 to skip relative goal entries, +1 to skip number of neighbors column
			idx = 1+self.state_dim_per_agent*(j+1)+np.arange(0,2,dtype=int)
			P[:,curr_idx,:] = x[:,idx]
			H[:,curr_idx] = torch.norm(x[:,idx], p=2, dim=1) - 2*self.param.r_agent
			curr_idx += 1 

		for j in range(no):
			idx = 1+self.state_dim_per_agent*(nn+1)+j*2+np.arange(0,2,dtype=int)	
			P[:,curr_idx,:] = x[:,idx]
			closest_point = torch_min_point_circle_rectangle(
				torch.zeros(2,device=self.device), 
				self.param.r_agent,
				-x[:,idx] - torch.tensor([0.5,0.5],device=self.device), 
				-x[:,idx] + torch.tensor([0.5,0.5],device=self.device))
			H[:,curr_idx] = torch.norm(closest_point, p=2, dim=1) - self.param.r_agent
			curr_idx += 1

		return P,H 

	def get_barrier_action(self,x):

		P,H = self.get_relative_positions_and_safety_functions(x)

		nd = x.shape[0] # number of data points in batch 
		nn = int(x[0,0].item()) # number of neighbors
		no = int((x.shape[1] - 1 - (nn+1)*self.state_dim_per_agent) / 2)  # number of obstacles 
		barrier = torch.zeros((len(x),self.action_dim_per_agent),device=self.device)

		for j in range(nn + no):
			barrier += self.get_barrier(P[:,j,:],H[:,j])

		return barrier


	def get_barrier(self,P,H):
		normP = torch.norm(P,p=2,dim=1)
		normP = normP.unsqueeze(1)
		normP = torch_tile(normP,1,P.shape[1])
		H = H.unsqueeze(1)
		H = torch_tile(H,1,P.shape[1])
		barrier = -1*self.param.b_gamma*torch.mul(torch.mul(torch.pow(H,-1),torch.pow(normP,-1)),P)
		return barrier


	def policy(self,x):

		A = np.empty((len(x),self.action_dim_per_agent))
		for i,x_i in enumerate(x):
			a_i = self(torch.tensor(x_i).float())
			a_i = a_i.detach().numpy()
			A[i,:] = a_i
		return A

	def empty(self,x):
		# batches are grouped by number of neighbors (i.e., each batch has data with the same number of neighbors)
		# x is a 2D tensor, where the columns are: relative_goal, relative_neighbors, ...
		if self.state_dim_per_agent == 2:
			num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
			num_obstacles = int((x.size()[1] - 3 - 2*num_neighbors)/2)
			rho_neighbors = self.model_neighbors.forward(x[:,3:3+2*num_neighbors])
			rho_obstacles = self.model_obstacles.forward(x[:,3+2*num_neighbors:])
			g = x[:,1:3]
		elif self.state_dim_per_agent == 4:
			num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
			num_obstacles = int((x.size()[1] - 5 - 4*num_neighbors)/2)
			rho_neighbors = self.model_neighbors.forward(x[:,5:5+4*num_neighbors])
			rho_obstacles = self.model_obstacles.forward(x[:,5+4*num_neighbors:])
			g = x[:,1:5]
		else:
			assert(False)

		x = torch.cat((rho_neighbors, rho_obstacles, g),1)
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x

	def get_adaptive_scaling(self,x,empty_action,barrier_action):
		P,H = self.get_relative_positions_and_safety_functions(x)
		adaptive_scaling = torch.ones(H.shape[0],device=self.device)
		# print('H',H)
		if not H.nelement() == 0:
			minH = torch.min(H,dim=1)[0]
			normb = torch.norm(barrier_action,p=2,dim=1)
			normpi = torch.norm(empty_action,p=2,dim=1)
			adaptive_scaling[minH < self.param.Delta_R] = torch.min(\
				torch.mul(normb,torch.pow(normpi,-1)),torch.ones(1,device=self.device))[0]
		return adaptive_scaling.unsqueeze(1)

	def __call__(self,x):

		# barrier_action = self.APF(x)
		barrier_action = self.get_barrier_action(x)
		empty_action = self.empty(x)
		empty_action = self.scale(empty_action, self.param.phi_max)
		adaptive_scaling = self.get_adaptive_scaling(x,empty_action,barrier_action)
		action = torch.mul(adaptive_scaling,empty_action)+barrier_action 
		action = self.scale(action, self.param.a_max)
		return action 

	def scale(self,action,max_action):
		inv_alpha = action.norm(p=2,dim=1)/max_action
		inv_alpha = torch.clamp(inv_alpha,min=1)
		inv_alpha = inv_alpha.unsqueeze(0).T
		inv_alpha = torch_tile(inv_alpha,1,2)
		action = action*inv_alpha.pow_(-1)
		return action 
