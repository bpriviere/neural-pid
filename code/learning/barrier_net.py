
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

		self.r_agent = param.r_agent
		self.r_obstacle = param.r_obstacle
		self.action_dim_per_agent = param.il_psi_network_architecture[-1].out_features
		self.state_dim_per_agent = param.il_phi_network_architecture[0].in_features
		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation
		self.a_max = param.a_max
		self.a_min = param.a_min 
		self.D_robot = param.D_robot
		self.D_obstacle = param.D_obstacle
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

		# batches are grouped by number of neighbors (i.e., each batch has data with the same number of neighbors)
		# x is a 2D tensor, where the columns are: relative_goal, relative_neighbors, ...

		num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
		num_obstacles = int((x.size()[1]-5 - 4*num_neighbors)/2)

		rho_neighbors = self.model_neighbors.forward(x[:,5:5+4*num_neighbors])
		rho_obstacles = self.model_obstacles.forward(x[:,5+4*num_neighbors:])
		g = x[:,1:3]
		x = torch.cat((rho_neighbors, rho_obstacles, g),1)

		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)

		x = torch.tanh(x) # x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((self.phi_max-self.phi_min)).float()+torch.tensor((self.phi_min)).float() #, x \in [amin,amax]
		return x

	def __call__(self,x):

		empty_action = self.empty(x)
		barrier_action = torch.zeros((len(x),self.action_dim_per_agent))
		nd = x.shape[0] # number of data points in batch 
		nn = int(x[0,0].item()) # number of neighbors
		no = int((x.shape[1] - 1 - (nn+1)*self.state_dim_per_agent) / 2)  # number of obstacles 

		for i, observation_i in enumerate(x):	

			# v_i = observation_i[2]

			for j in range(nn):
				# j+1 to skip relative goal entries, +1 to skip number of neighbors column
				idx = 1+self.state_dim_per_agent*(j+1)+np.arange(0,self.state_dim_per_agent,dtype=int)
				relative_neighbor = observation_i[idx]
				p_ij = -1*relative_neighbor[0:2]
				v_ij = -1*relative_neighbor[2:]
				barrier_action[i,:] += self.get_robot_barrier(p_ij,v_ij)

			for j in range(no):
				# pass 
				idx = 1 + self.state_dim_per_agent*(nn+1)+np.arange(0,2,dtype=int)
				p_ij = observation_i[idx]
				barrier_action[i,:] += self.get_obstacle_barrier(p_ij)

		# scale actions 
		action = empty_action + barrier_action 
		action = torch.tanh(action) # action \in [-1,1]
		action = (action+1.)/2.*torch.tensor((self.a_max-self.a_min)).float()+torch.tensor((self.a_min)).float() # action \in [amin,amax]
		return action 

	def get_robot_barrier(self,dp,dv):
		
		"""
		this is a barrier function (works like artificial potential function) 

		dp = p^i - p^j
		dv = v^i - v^j
		"""
		
		# h_ij = np.sqrt(4*self.a_max*(np.linalg.norm(dp) - self.Ds)) \
		# 	+ np.matmul(dv.T, dp)/np.linalg.norm(dp)

		h_ij = np.linalg.norm(dp) - self.D_robot
		if h_ij > 0:
			return self.b_gamma/np.power(h_ij,self.b_exph)*dp 
		else:
			return self.b_gamma/np.power(-1*h_ij,self.b_exph)*-1*dp 

	def get_obstacle_barrier(self,dp):

		h_ij = self.get_min_dist(dp)
		# h_ij = min_dist - self.D_obstacle
		if h_ij > 0:
			return self.b_gamma/np.power(h_ij,self.b_exph)*dp 
		else:
			return self.b_gamma/np.power(-1*h_ij,self.b_exph)*-1*dp 

	def get_min_dist(self,dp):

		if False:
			# TEMP 
			d2 = np.linalg.norm(dp) - self.D_obstacle

		else: 

			def line(p1, p2):
				A = (p1[1] - p2[1])
				B = (p2[0] - p1[0])
				C = (p1[0]*p2[1] - p2[0]*p1[1])
				return A, B, -C

			def intersection(L1, L2):
				D  = L1[0] * L2[1] - L1[1] * L2[0]
				Dx = L1[2] * L2[1] - L1[1] * L2[2]
				Dy = L1[0] * L2[2] - L1[2] * L2[0]
				if D != 0:
					x = Dx / D
					y = Dy / D
					return np.array([x,y])
				else:
					return np.array([False,False])

			# dp = pi - pj 
			# shift coordinate st pj = [0,0] (center of square obstacle)
			# line_to_agent: line from pj to pi
			# obstacle boundaries: [0,0] +- r_obstacle*[1,1] 
			# d1 is the length from center of square to the intersection point on the square
			# d2 is the length from the agent boundary to the d1
			# norm p is the length from the center of the agent to the center of the obstacle 

			dp = dp.numpy()
			line_to_agent = line([0,0],[dp[0],dp[1]])

			square_lines = []
			square_lines.append(line(
				[0-self.r_obstacle,0-self.r_obstacle],
				[0-self.r_obstacle,0+self.r_obstacle]))
			square_lines.append(line(
				[0-self.r_obstacle,0+self.r_obstacle],
				[0+self.r_obstacle,0+self.r_obstacle]))
			square_lines.append(line(
				[0+self.r_obstacle,0+self.r_obstacle],
				[0+self.r_obstacle,0-self.r_obstacle]))
			square_lines.append(line(
				[0+self.r_obstacle,0-self.r_obstacle],
				[0-self.r_obstacle,0-self.r_obstacle]))

			d1 = np.inf 
			for square_line in square_lines:
				r = intersection(line_to_agent,square_line)
				norm_r = np.linalg.norm(r)

				if r.any() and norm_r < d1:
					d1 = norm_r 

			d2 = np.linalg.norm(dp) - d1 - self.r_agent

		return d2




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