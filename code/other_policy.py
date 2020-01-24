

import numpy as np 
import cvxpy as cp
from cvxpy.atoms.norm_inf import norm_inf
from collections import namedtuple


from utilities import torch_tile, min_dist_circle_rectangle, torch_min_point_circle_rectangle
import torch 

# motion planning 

class Empty_Net_wAPF():

	def __init__(self,param,env,empty_net):

		self.empty_net = empty_net

		self.action_dim_per_agent = param.il_psi_network_architecture[-1].out_features
		self.state_dim_per_agent = param.il_phi_network_architecture[0].in_features
		
		self.a_max = param.a_max
		self.a_min = param.a_min
		# self.a_noise = param.a_noise

		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation

		self.param = param 

		self.device = torch.device('cpu')


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
			H[:,curr_idx] = torch.max(torch.norm(x[:,idx], p=2, dim=1) - 2*self.param.r_agent, torch.zeros(1,device=self.device))
			curr_idx += 1 

		for j in range(no):
			idx = 1+self.state_dim_per_agent*(nn+1)+j*2+np.arange(0,2,dtype=int)	
			P[:,curr_idx,:] = x[:,idx]
			closest_point = torch_min_point_circle_rectangle(
				torch.zeros(2,device=self.device), 
				self.param.r_agent,
				-x[:,idx] - torch.tensor([0.5,0.5],device=self.device), 
				-x[:,idx] + torch.tensor([0.5,0.5],device=self.device))
			# H[:,curr_idx] = torch.norm(closest_point, p=2, dim=1) - self.param.r_agent
			H[:,curr_idx] = torch.max(torch.norm(closest_point, p=2, dim=1) - self.param.r_agent, torch.zeros(1,device=self.device))
			curr_idx += 1

		return P,H 

	def get_barrier_action(self,x):

		P,H = self.get_relative_positions_and_safety_functions(x)

		nd = x.shape[0] # number of data points in batch 
		nn = int(x[0,0].item()) # number of neighbors
		no = int((x.shape[1] - 1 - (nn+1)*self.state_dim_per_agent) / 2)  # number of obstacles 
		barrier = torch.zeros((len(x),self.action_dim_per_agent))

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
			x_i = torch.Tensor(x_i)
			empty_action = self.empty_net(x_i)
			empty_action = self.scale(empty_action, self.param.phi_max)
			barrier_action = self.get_barrier_action(x_i)
			adaptive_scaling = self.get_adaptive_scaling(x_i,empty_action,barrier_action)
			a_i = adaptive_scaling*empty_action + barrier_action
			a_i = self.scale(a_i,self.param.a_max)
			a_i = a_i.detach().numpy()
			A[i,:] = a_i
		return A


	def scale(self,action,value):
		# scale 
		inv_alpha = action.norm(p=float('inf'),dim=1)/value
		inv_alpha = torch.clamp(inv_alpha,min=1)
		inv_alpha = inv_alpha.unsqueeze(0).T
		inv_alpha = torch_tile(inv_alpha,1,2)
		action = action*inv_alpha.pow_(-1)
		return action 

	def get_adaptive_scaling(self,x,empty_action,barrier_action):
		P,H = self.get_relative_positions_and_safety_functions(x)
		adaptive_scaling = torch.ones(H.shape[0])
		if not H.nelement() == 0:
			minH = torch.min(H,dim=1)[0]
			normb = torch.norm(barrier_action,p=2,dim=1)
			normpi = torch.norm(empty_action,p=2,dim=1)
			adaptive_scaling[minH < self.param.Delta_R] = torch.min(torch.mul(normb,torch.pow(normpi,-1)),torch.Tensor(1))[0]
		return adaptive_scaling.unsqueeze(1)




# other
# consensus policies 

class ZeroPolicy:
	def __init__(self,env):
		self.env = env
	def policy(self,state):
		return np.zeros((self.env.m))

class LCP_Policy:
	# linear consensus protocol
	def __init__(self,env):
		self.env = env

	def policy(self,observation):
		a = np.zeros((self.env.m))
		dt = self.env.times[self.env.time_step+1] - self.env.times[self.env.time_step]
		n_neighbors = self.env.n_neighbors
		agent_memory = self.env.agent_memory
		for agent in self.env.agents:
			# observation_i = {s^j - s^i} \forall j in N^i
			
			idx = np.arange(agent_memory,len(observation[agent.i]),agent_memory)

			# print(observation[agent.i])
			# print(observation[agent.i][idx])
			# exit()

			a[agent.i] = sum(observation[agent.i][idx])*dt
		return a

class WMSR_Policy:

	def __init__(self,env):
		self.env = env

	# weighted mean subsequence reduced 
	def policy(self,observation):
		f = self.env.n_malicious
		a = np.zeros((self.env.m))
		dt = self.env.times[self.env.time_step+1] - self.env.times[self.env.time_step] 
		n_neighbors = self.env.n_neighbors

		for agent in self.env.agents: 

			x_i = agent.x 

			x_js = np.zeros((n_neighbors))
			count = 0 
			for agent_j in self.env.agents:
				if not agent_j.i == agent.i:
					x_js[count] = observation[agent.i][agent_j.i][0]
					count += 1
			x_js = x_js + x_i


			x_js_sorted = np.sort( np.array(x_js)) 

			ng = sum(x_i<x_js)
			nl = sum(x_i>x_js)
		
			R = []
			if ng >= f:
				# add f largest values to R 
				for i in range(f):
					R.append(x_js_sorted[-(i+1)])
			else: 
				# add all values larger than x_i
				for x_j in x_js: 
					if x_j > x_i:
						R.append(x_j)

			if nl >= f:
				# add f smallest values to R
				for i in range(f):
					R.append(x_js_sorted[i])
			else:
				# add all values smaller than x_i
				for x_j in x_js:
					if x_j < x_i:
						R.append(x_i)
			
			# R = np.array(R)

			count = 0
			summ = 0 #x_i
			for x_j in x_js:
				if not x_j in R:
					count += 1
					summ += x_j-x_i
			if count != 0:
				a[agent.i] = summ/count
			else:
				a[agent.i] = 0.0 

			# print(observation[agent.i])
			# print(x_i)
			# print(x_js)
			# print(ng)
			# print(nl)
			# print(R)
			# exit()

		return a*dt




# control barrier function as implemented by Ames 2017
# static obstacles not implemented  
class CBF:

	def __init__(self,param,env):
		self.env = env
		self.param = param
		self.Ds = param.r_safe
		self.alpha = param.a_max 
		self.state_dim_per_agent = env.state_dim_per_agent
		self.action_dim_per_agent = env.action_dim_per_agent
		self.gamma = param.b_gamma 

	# control barrier function 
	def policy(self,observations):

		A = np.zeros((len(observations),self.action_dim_per_agent))
		
		print('t: ', self.env.times[self.env.time_step])
		
		for agent_i in self.env.agents:

			observation = observations[agent_i.i]
			relative_goal = observation[0:self.state_dim_per_agent]
			relative_neighbors = observation[self.state_dim_per_agent:]
			n_neighbor = int(len(relative_neighbors)/self.state_dim_per_agent)

			# calculate nominal controller
			a_nom = self.param.cbf_kp*relative_goal[0:2] + self.param.cbf_kv*relative_goal[2:] # [pgx - pix, pgy - piy]
			scale = self.alpha/np.max(np.abs(a_nom))
			if scale < 1:
				a_nom = scale*a_nom 

			# print()
			# print('n_neighbor:',n_neighbor)
			# print('i: ', agent_i.i)	
			# print('observation: ', observation)
			# print('relative_goal: ', relative_goal)
			# print('vi: ', relative_goal[2:])
			# print('relative_neighbors: ', relative_neighbors)
			# print('sg: ', agent_i.s_g)
			# print('agent_i.p: ', agent_i.p)
			# print('a_nom: ', a_nom)
			# print()
			# exit()


			# CVX
			a_i = cp.Variable(self.action_dim_per_agent)
			v_i = -1*relative_goal[2:]
			dt = self.param.sim_dt
			constraints = [] 

			if not n_neighbor == 0:
				for j in range(n_neighbor):
					rn_idx = self.state_dim_per_agent*j+np.arange(0,self.state_dim_per_agent,dtype=int)
					
					p_ij = -1*relative_neighbors[rn_idx[0:2]]
					v_ij = -1*relative_neighbors[rn_idx[2:]]

					A_ij = -p_ij.T
					b_ij = 1/2*self.get_b_ij(p_ij,v_ij)
					
					constraints.append(A_ij@a_i <= b_ij) 

			# acceleration and velocity limits 
			constraints.append(norm_inf(a_i) <= self.alpha) 
			constraints.append(norm_inf(v_i+a_i*dt) <= self.param.v_max)

			obj = cp.Minimize( cp.sum_squares( a_i - a_nom)) 
			prob = cp.Problem( obj, constraints) 

			# print('Solving...')

			try:
				prob.solve(verbose=False, solver = cp.GUROBI)
				# prob.solve(verbose=True)

				if prob.status in ["optimal"]:
					a_i = np.array(a_i.value)
				else:
					# do nothing 
					# a_i = 0*a_nom

					# brake
					# a_i = -self.alpha*relative_goal[0:2]

					# backup 
					a_i = -self.alpha*v_i/np.linalg.norm(v_i)


			except Exception as e:
				print(e)
				# do nothing 
				# a_i = 0*a_nom
				
				# brake
				# a_i = -self.alpha*relative_goal[0:2]

				# backup 
				a_i = -self.alpha*v_i/np.linalg.norm(v_i)

			A[agent_i.i,:] = a_i + self.param.cbf_noise*np.random.normal(size=(1,2))
			# A[agent_i.i,:] = a_i 

		# exit()
		# print('A: ',A)
		return A 

	def get_h_ij(self,dp,dv):
		h_ij = np.sqrt(4*self.alpha*(np.linalg.norm(dp) - self.Ds)) \
			+ np.matmul(dv.T, dp)/np.linalg.norm(dp)
		return h_ij


	def get_b_ij(self,dp,dv):
		# this is linear coefficient in Ax <= b equation (not barrier function)
		h_ij = self.get_h_ij(dp,dv)
		delta_vTp = np.matmul(dv.T, dp)
		b_ij = self.gamma * np.power(h_ij,3) * np.linalg.norm(dp) \
			- np.power(delta_vTp,2)/np.power(np.linalg.norm(dp),2) \
			+ (2*self.alpha*delta_vTp) \
			/ np.sqrt(4*self.alpha*(np.linalg.norm(dp)-self.Ds)) \
			+ np.power(np.linalg.norm(dv),2)
		return b_ij 

	def get_neighborhood_dist(self):
		pass



# artifical potential function with linear goal 
class APF:

	def __init__(self,param,env):
		self.env = env
		self.param = param
		self.D_robot = param.D_robot
		self.D_obstacle = param.D_obstacle
		self.alpha = param.a_max 
		self.state_dim_per_agent = env.state_dim_per_agent
		self.action_dim_per_agent = env.action_dim_per_agent
		self.b_gamma = param.b_gamma 
		self.r_obstacle = param.r_obstacle
		self.r_agent = param.r_agent
		self.b_exph = param.b_exph
		self.a_max = param.a_max
		self.a_min = param.a_min

	# control barrier function 
	def policy(self,observations):
		
		# print('t: ', self.env.times[self.env.time_step])

		ni = len(observations) # number of agents 
		A = np.empty((ni,self.action_dim_per_agent))
		
		for i, observation_i in enumerate(observations):

			nn = int(observation_i[0]) # number of neighbors
			no = int((len(observation_i) - 1 - (nn+1)*self.state_dim_per_agent) / 2)  # number of obstacles 

			# calculate nominal controller
			rg_idx = 1+np.arange(0,self.state_dim_per_agent)
			relative_goal = observation_i[rg_idx]
			a_nom = self.param.cbf_kp*relative_goal[0:2] + self.param.cbf_kv*relative_goal[2:] # [pgx - pix, pgy - piy]
			scale = self.alpha/np.max(np.abs(a_nom))
			if scale < 1:
				a_nom = scale*a_nom

			# get barrier action
			a_barrier = np.zeros(self.action_dim_per_agent)

			# neighbor barrier 
			for j in range(nn):
				# j+1 to skip relative goal entries, +1 to skip number of neighbors column
				idx = 1+self.state_dim_per_agent*(j+1)+np.arange(0,self.state_dim_per_agent,dtype=int)
				relative_neighbor = observation_i[idx]
				p_ij = -1*relative_neighbor[0:2]
				v_ij = -1*relative_neighbor[2:]
				a_barrier += self.get_robot_barrier(p_ij,v_ij)

			# obstacle barrier 
			for j in range(no):
				# pass 
				idx = 1 + self.state_dim_per_agent*(nn+1)+np.arange(0,2,dtype=int)
				p_ij = -1*observation_i[idx]
				a_barrier += self.get_obstacle_barrier(p_ij)

			# add 
			a_i = a_barrier + a_nom 

			# scale: 
			if False:
				a_i = np.tanh(a_i) # action \in [-1,1]
				a_i = (a_i+1.)/2.*(self.a_max-self.a_min)+self.a_min # action \in [amin,amax]

			else:
				alpha = self.a_max / max(np.abs(a_i)) 
				if alpha < 1:
					a_i = a_i*alpha 

			# A[i,:] = a_i + self.param.cbf_noise*np.random.normal(size=(1,2))
			A[i,:] = a_i 

		# exit()
		# print('A: ',A)
		return A 

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

		if True:
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


