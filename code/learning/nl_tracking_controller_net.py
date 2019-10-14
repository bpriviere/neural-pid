
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np 


class NL_EL_Tracking_Controller_Net(nn.Module):
	"""
	Nonlinear Euler Lagrangian Tracking Controller

	pi(s) = a
	where last layer:
	a = M dotdot{q}_r + C dot{q}_r - K (dot{q} -dot{q}_r) + G(q)
	dot{q}_r = dot{q}_d - Lbda (q_d - q)
	"""

	def __init__(self, robot_env, K, Lbda, nn_architecture, ):
		
		super(Nl_Tracking_Controller_Net, self).__init__()

		self.K = K # drives dot{q} to dot{q}_r 
		self.Lbda = Lbda # drives q to q_r 
		self.M = robot_env.M 
		self.C = robot_env.C 
		self.G = robot_env.G 
		
		

		self.fc1 = nn.Linear(state_dim, 16)
		self.fc2 = nn.Linear(16, 16)
		self.fc3 = nn.Linear(16, state_dim)
		
		self.n = state_dim
		self.m = action_dim

	def evalNN(self, x):
		x = torch.from_numpy(np.array(x,ndmin = 2)).float()
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x = self.fc3(x)
		return x

	def forward(self, x):
		# input: 
		# 	x, nd array, (n,)
		# output:
		# 	a, nd array, (m,1)

		# batch input: 
		# 	x, torch tensor, (ndata,n)
		# 	a, torch tensor, (ndata,m)

		state = torch.from_numpy(np.array(x,ndmin = 2)).float()
		ref_state = self.evalNN(x)

		# error (proportional and derivative)
		error = state-ref_state
		ep = error[:,0:int(self.n/2)]
		ed = error[:,int(self.n/2):]
		
		# gain matrix 
		Kp = torch.tensor(self.Kp*np.ones((self.m,int(self.n/2)))).float()
		Kd = torch.tensor(self.Kd*np.ones((self.m,int(self.n/2)))).float()

		# PD control 
		a = (torch.mm(Kp,ep.T) + torch.mm(Kd,ed.T)).T

		return a

	def policy(self,state):
		action = self(torch.from_numpy(state).float())
		action = np.squeeze(action.detach().numpy())
		return action

	def get_kp(self,x):
		return self.Kp

	def get_kd(self,x):
		return self.Kd

	def get_ref_state(self,x):
		x = self.evalNN(x)
		x = x.detach().numpy()
		return x