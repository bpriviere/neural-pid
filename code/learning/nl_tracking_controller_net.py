
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
	s = [q;dot{q}]
	sd = [qd;dot{qd}]
	where last layer:
	a = M dotdot{q}_r + C dot{q}_r - K (dot{q} -dot{q}_r) + G(q)
	dot{q}_r = dot{q}_d - Lbda (q_d - q)
	"""

	def __init__(self, robot_env, K, Lbda, layers, activation):
		
		super(Nl_Tracking_Controller_Net, self).__init__()

		self.K = torch.tensor(K) # drives dot{q} to dot{q}_r 
		self.Lbda = torch.tensor(Lbda) # drives q to q_r 
		self.M = robot_env.M 
		self.C = robot_env.C 
		self.G = robot_env.G 
		self.f = robot_env.f
		
		self.layers = layers
		self.activation = activation

		self.state_dim = layers[0].in_features
		self.action_dim = layers[-1].out_features

	def forward(self, x):
		# input: 
		# 	x, nd array, (n,)
		# output:
		# 	a, nd array, (m,1)

		# batch input: 
		# 	x, torch tensor, (ndata,n)
		# 	a, torch tensor, (ndata,m)

		# save input state
		s = x 

		# get 'reference' trajectory by propagating thru network 
		for layer in layers[:-1]:
			x = self.activation(layer(x))
		qd = layers[-1](x)[0:state_dim/2]
		dot_qd = layers[-1](x)[state_dim/2:state_dim]
		dotdot_qd = layers[-1](x)[state_dim:]

		# extract current state 
		q = s[0:state_dim/2]
		dot_q = s[state_dim/2:]

		# composite variable
		dot_qr = dot_qd - self.Lbda@(qd - q)
		dotdot_qr = dotdot_qd - self.Lbda@(dot_qd - dot_q)

		# system
		M = self.M(q,q_dot) 
		C = self.C(q)
		G = self.G(q)

		# control law
		a = M @ dotdot_qr + C @ dot_qr + G - self.K @ (dot_q - dot_qr)
		return a

	def policy(self,state):
		action = self(torch.from_numpy(state).float())
		action = np.squeeze(action.detach().numpy())
		return action