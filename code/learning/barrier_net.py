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
from utilities import torch_tile, min_dist_circle_rectangle


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
		self.circle_obstacles_on = param.circle_obstacles_on
		self.device = torch.device('cpu')

	def to(self, device):
		self.device = device
		self.model_neighbors.to(device)
		self.model_obstacles.to(device)
		return super().to(device)

	def policy(self,x,transformations):
		# this fnc is only used at runtime 

		# inputs observation from all agents...
		# outputs policy for all agents

		A = np.empty((len(x),self.action_dim_per_agent))
		for i,x_i in enumerate(x):

			R = transformations[i][0]

			a_i = self(torch.Tensor(x_i))
			a_i = a_i.detach().numpy()
			a_i = np.matmul(R.T,a_i.T).T
			A[i,:] = a_i

			# a_i = self(torch.Tensor([x_i]))
			# A[i,:] = a_i.detach().numpy()
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


	def scale_empty(self,x):
		x = torch.tanh(x) # x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((self.phi_max-self.phi_min)).float()+torch.tensor((self.phi_min)).float() #, x \in [amin,amax]
		return x


	def __call__(self,x):
		# this fnc is used for training 

		empty_action = self.empty(x)
		empty_action = self.scale(empty_action, self.phi_min, self.phi_max)
		barrier_action = self.APF(x)
		action = empty_action+barrier_action 
		action = self.scale(action, self.a_min, self.a_max)
		return action 

		# empty_action = self.empty(x)
		# barrier_action = self.APF(x)
		# action = empty_action + barrier_action 
		# return action

		# empty_action = self.scale(self.empty(x),self.phi_max)
		# barrier_action = self.APF(x)
		# action = empty_action + barrier_action
		# return action

		# empty_action = self.scale_empty(self.empty(x))
		# return empty_action

	def APF(self,x):
		nd = x.shape[0] # number of data points in batch 
		nn = int(x[0,0].item()) # number of neighbors
		no = int( (x.shape[1] - 1 - 2 - nn*self.state_dim_per_agent) / 2)  # number of obstacles 
			
		closest_barrier_mode_on = True
		if closest_barrier_mode_on:
			# this implementation uses only the closest barrier 
			barrier_action = torch.zeros((len(x),self.action_dim_per_agent), device=self.device)

			# The KD-Tree output is sorted, so we only need to check if we are closer to obstacle or neighbor
			# TODO: 
			#  1. switch numpy -> torch
			#  2. switch to full on-device cuda computation

			# TODO: the KD-Tree outputs data is sorted, so technically no need to loop over everything
			min_neighbor_dist = np.repeat(np.Inf, len(x))
			min_neighbor_p = np.empty((len(x),2))
			min_neighbor_mode = np.repeat(0, len(x))
			for j in range(nn):
				# j+1 to skip relative goal entries, +1 to skip number of neighbors column
				idx = 1+2+self.state_dim_per_agent*j+np.arange(0,self.state_dim_per_agent,dtype=int)
				relative_neighbor = x[:,idx].to(torch.device('cpu')).numpy()
				P_i = -1*relative_neighbor[:,0:2] # pi - pj
				dist = np.linalg.norm(P_i, axis=1) - self.r_agent
				predicate = dist < min_neighbor_dist
				min_neighbor_p[predicate] = P_i[predicate]
				min_neighbor_dist[predicate] = dist[predicate]
				min_neighbor_mode[predicate] = 1

			for j in range(no):
				idx = 1+2+self.state_dim_per_agent*nn+j*2+np.arange(0,2,dtype=int)
				P_i = -1*x[:,idx].to(torch.device('cpu')).numpy() # in nd x state_dim_per_agent
				closest, dist = min_dist_circle_rectangle(
					np.zeros(2), self.r_agent,
					P_i - np.array([0.5,0.5]), P_i + np.array([0.5,0.5]))

				predicate = dist < min_neighbor_dist
				min_neighbor_p[predicate] = closest[predicate]
				min_neighbor_dist[predicate] = dist[predicate]
				min_neighbor_mode[predicate] = 2

			# predicate = np.logical_and(min_neighbor_mode==1, min_neighbor_dist < self.r_agent + 0.15)
			predicate = min_neighbor_mode==1
			barrier_action[predicate] += torch.from_numpy(self.get_robot_barrier(min_neighbor_p)[predicate]).float().to(self.device)

			# predicate = np.logical_and(min_neighbor_mode==2, min_neighbor_dist < self.r_agent + 0.15)
			predicate = min_neighbor_mode==2
			barrier_action[predicate] += torch.from_numpy(self.get_obstacle_barrier_square(min_neighbor_p)[predicate]).float().to(self.device)

		else:
			# this implementation uses all barriers
			barrier_action = torch.zeros((len(x),self.action_dim_per_agent))
			# print('Neighbors')
			for j in range(nn):
				# j+1 to skip relative goal entries, +1 to skip number of neighbors column
				idx = 1+self.state_dim_per_agent*(j+1)+np.arange(0,self.state_dim_per_agent,dtype=int)
				relative_neighbor = x[:,idx].numpy()
				P_i = -1*relative_neighbor[:,0:2] # pi - pj
				A_i = self.get_robot_barrier(P_i)
				barrier_action += torch.from_numpy(A_i).float()

			# print('Obstacles')
			for j in range(no):
				idx = 1+self.state_dim_per_agent*(nn+1)+j*2+np.arange(0,2,dtype=int)
				P_i = -1*x[:,idx].numpy() # in nd x state_dim_per_agent
				A_i = self.get_obstacle_barrier(P_i)
				barrier_action += torch.from_numpy(A_i).float()

		return barrier_action 


	def scale(self,x, min_value, max_value):
		x = torch.tanh(x) # x \in [-1,1]
		x = (x+1.)/2.*torch.tensor((max_value-min_value)).float()+torch.tensor((min_value)).float() #, x \in [amin,amax]
		return x
		# inv_alpha = action.norm(p=float('inf'),dim=1)/self.a_max 
		# inv_alpha = torch.clamp(inv_alpha,min=1)
		# inv_alpha = inv_alpha.unsqueeze(0).T
		# inv_alpha = torch_tile(inv_alpha,1,2)
		# action = action*inv_alpha.pow_(-1)
		# return action 


	def get_robot_barrier(self,P):
		H = np.linalg.norm(P,axis=1) - self.D_robot
		H = np.reshape(H,(len(H),1))
		H = np.tile(H,(1,np.shape(P)[1]))
		normP = np.linalg.norm(P,axis=1)
		normP = np.reshape(normP,(len(normP),1))
		normP = np.tile(normP,(1,np.shape(P)[1]))
		return self.b_gamma*np.multiply(np.multiply(np.power(normP,-1),np.power(H,-1*self.b_exph)),P)


	def get_obstacle_barrier(self,P):
		H = np.linalg.norm(P,axis=1) - self.D_obstacle
		H = np.reshape(H,(len(H),1))
		H = np.tile(H,(1,np.shape(P)[1]))
		normP = np.linalg.norm(P,axis=1)
		normP = np.reshape(normP,(len(normP),1))
		normP = np.tile(normP,(1,np.shape(P)[1]))
		return self.b_gamma*np.multiply(np.multiply(np.power(normP,-1),np.power(H,-1*self.b_exph)),P)


	def get_obstacle_barrier_square(self,P):
		H = np.linalg.norm(P,axis=1) - self.r_agent
		H = np.reshape(H,(len(H),1))
		H = np.tile(H,(1,np.shape(P)[1]))
		normP = np.linalg.norm(P,axis=1)
		normP = np.reshape(normP,(len(normP),1))
		normP = np.tile(normP,(1,np.shape(P)[1]))
		return self.b_gamma*np.multiply(np.multiply(np.power(normP,-1),np.power(H,-1*self.b_exph)),P)

	# old fncs 
	# def get_robot_barrier(self,dp,dv):
	# 	norm_p = np.linalg.norm(dp)
	# 	h_ij = norm_p - self.D_robot
	# 	return self.b_gamma/np.power(h_ij,self.b_exph)*dp/norm_p

	# def get_obstacle_barrier(self,dp):
	# 	norm_p = np.linalg.norm(dp)

	# 	if self.circle_obstacles_on:
	# 		h_ij = norm_p - self.D_obstacle

	# 	else:

	# 		# dp = pi - pj 
	# 		# shift coordinate st pj = [0,0] (center of square obstacle)
	# 		# line_to_agent: line from pj to pi
	# 		# obstacle boundaries: [0,0] +- r_obstacle*[1,1] 
	# 		# d1 is the length from center of square to the intersection point on the square
	# 		# d2 is the length from the agent boundary to the d1
	# 		# norm p is the length from the center of the agent to the center of the obstacle 

	# 		def line(p1, p2):
	# 			A = (p1[1] - p2[1])
	# 			B = (p2[0] - p1[0])
	# 			C = (p1[0]*p2[1] - p2[0]*p1[1])
	# 			return A, B, -C

	# 		def intersection(L1, L2):
	# 			D  = L1[0] * L2[1] - L1[1] * L2[0]
	# 			Dx = L1[2] * L2[1] - L1[1] * L2[2]
	# 			Dy = L1[0] * L2[2] - L1[2] * L2[0]
	# 			if D != 0:
	# 				x = Dx / D
	# 				y = Dy / D
	# 				return np.array([x,y])
	# 			else:
	# 				return np.array([False,False])

	# 		dp = dp.numpy()
	# 		line_to_agent = line([0,0],[dp[0],dp[1]])

	# 		square_lines = []
	# 		square_lines.append(line(
	# 			[0-self.r_obstacle,0-self.r_obstacle],
	# 			[0-self.r_obstacle,0+self.r_obstacle]))
	# 		square_lines.append(line(
	# 			[0-self.r_obstacle,0+self.r_obstacle],
	# 			[0+self.r_obstacle,0+self.r_obstacle]))
	# 		square_lines.append(line(
	# 			[0+self.r_obstacle,0+self.r_obstacle],
	# 			[0+self.r_obstacle,0-self.r_obstacle]))
	# 		square_lines.append(line(
	# 			[0+self.r_obstacle,0-self.r_obstacle],
	# 			[0-self.r_obstacle,0-self.r_obstacle]))

	# 		d_square_to_intersection = np.inf 
	# 		for square_line in square_lines:
	# 			r = intersection(line_to_agent,square_line)
	# 			norm_r = np.linalg.norm(r)

	# 			if r.any() and norm_r < d_square_to_intersection:
	# 				d_square_to_intersection = norm_r 

	# 		h_ij = np.linalg.norm(dp) - d_square_to_intersection - self.r_agent


	# 	return self.b_gamma/np.power(h_ij,self.b_exph)*dp/norm_p