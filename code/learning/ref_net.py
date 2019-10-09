import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from param import param 
from numpy import squeeze, array,arange, linspace
import numpy as np 


class Ref_Net(nn.Module):
	"""
	neural net to state_ref
	"""
	def __init__(self, input_layer, Kp, Kd):
		super(Ref_Net, self).__init__()
		self.Kp = Kp
		self.Kd = Kd
		self.fc1 = nn.Linear(input_layer, 16)
		self.fc2 = nn.Linear(16, 16)
		self.fc3 = nn.Linear(16, input_layer)

	def evalNN(self, x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x = self.fc3(x)
		return x

	def forward(self, x):
		state = torch.from_numpy(array(x,ndmin = 2)).float()
		x = self.evalNN(x)
		ref_state = x
		error = state-ref_state
		x = self.Kp[0]*error[:,0] + self.Kp[1]*error[:,1] + \
			self.Kd[0]*error[:,2] + self.Kd[0]*error[:,3] 
		x = x.reshape((len(x),1))
		return x

	def policy(self,state):
		action = self(torch.from_numpy(state).float())
		action = squeeze(action.detach().numpy())
		return action

	def get_kp(self,x):
		return self.Kp

	def get_kd(self,x):
		return self.Kd

	def get_ref_state(self,x):
		x = self.evalNN(x)
		x = x.detach().numpy()
		return x