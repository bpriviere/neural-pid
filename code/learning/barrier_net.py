
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
			param.il_phi_obs_network_architecture,
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
		self.Ds = param.r_safe
		self.b_gamma = param.b_gamma 
		self.b_exph = param.b_exph
		self.phi_min = param.phi_min
		self.phi_max = param.phi_max

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

		x = torch.tanh(x) # x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((self.phi_max-self.phi_min)).float()+torch.tensor((self.phi_min)).float() #, x \in [amin,amax]
		return x

	# def __call__(self,x):

	# 	empty_action = self.empty(x)
	# 	barrier_action = torch.zeros((len(x),self.action_dim_per_agent))

	# 	nd = len(x) # number of data 
	# 	nn = int(len(x[0])/(self.state_dim_per_agent)-1) # number of neighbors in this batch 

	# 	P_ij = np.zeros((nd,nn*2))
	# 	V_ij = np.zeros((nd,nn*2))

	# 	for j in range(nn):
	# 		col_idx = j*4 + np.arange(0,2,dtype=int)
	# 		P_ij[:,col_idx] = x[:,col_idx + 4]
	# 		V_ij[:,col_idx] = x[:,col_idx + 6]

	# 	print(P_ij[0,:])
	# 	print(V_ij[0,:])
	# 	print(x[0,:])
	# 	exit()

	# 	H = 

	def __call__(self,x):

		empty_action = self.empty(x)
		barrier_action = torch.zeros((len(x),self.action_dim_per_agent))
		nd = x.shape[0] # number of data points in batch 
		nn = x[0,0] # number of neighbors
		no = (x.shape[1] - 1 - (nn+1)*self.state_dim_per_agent) / 2  # number of obstacles 

		for i, observation_i in enumerate(x):	

			v_i = observation_i[2]

			for j in range(nn):
				# j+1 to skip relative goal entries, +1 to skip number of neighbors column
				idx = 1+self.state_dim_per_agent*(j+1)+np.arange(0,self.state_dim_per_agent,dtype=int)
				relative_neighbor = observation_i[idx]
				p_ij = -1*relative_neighbor[0:2]
				v_ij = -1*relative_neighbor[2:]
				barrier_action[i,:] += self.get_robot_barrier(p_ij,v_ij)*p_ij

			for j in range(no):
				pass 

		# # scale actions 
		action = empty_action + barrier_action 
		action = torch.tanh(action) # action \in [-1,1]
		action = (action+1.)/2.*torch.tensor((self.a_max-self.a_min)).float()+torch.tensor((self.a_min)).float() # action \in [amin,amax]
		return action 

	def get_B_i():
		pass

	def get_robot_barrier(self,dp,dv):
		
		"""
		this is a barrier function (works like artificial potential function) 

		dp = p^i - p^j
		dv = v^i - v^j
		"""
		h_ij = self.get_h_ij(dp,dv)
		return self.b_gamma/np.power(h_ij,self.b_exph)

	def get_obstacle_barrier(self,dp,dv):




		h_ij = self.get_h_ij(min_dist,projected_velocity)
		return self.b_gamma/np.power(h_ij,self.b_exph)

	def get_h_ij(self,dp,dv):
		# h_ij = np.sqrt(4*self.a_max*(np.linalg.norm(dp) - self.Ds)) \
		# 	+ np.matmul(dv.T, dp)/np.linalg.norm(dp)
		h_ij = np.linalg.norm(dp) - self.Ds
		# h_ij = np.sqrt(np.matmul(dp.T,dp)) - self.Ds
		return h_ij



