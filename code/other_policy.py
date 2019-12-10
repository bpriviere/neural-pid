

import numpy as np 
import cvxpy as cp
from cvxpy.atoms.norm_inf import norm_inf
from collections import namedtuple

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


class CBF:

	def __init__(self,param,env):
		self.env = env
		self.Ds = 2.75*param.r_agent
		self.alpha = param.a_max 
		self.state_dim_per_agent = env.state_dim_per_agent
		self.action_dim_per_agent = env.action_dim_per_agent
		self.gamma = param.b_gamma 

	# control barrier function 
	def policy(self,observations):

		A = np.zeros((len(observations),self.action_dim_per_agent))
		
		for agent_i in self.env.agents:

			observation = observations[agent_i.i]
			relative_goal = observation[0:self.state_dim_per_agent]
			relative_neighbors = observation[self.state_dim_per_agent:]
			n_neighbor = int(len(relative_neighbors)/self.state_dim_per_agent)

			# calculate nominal controller
			a_nom = 0.01*relative_goal[0:2] # + 0.1 * relative_goal[2:] # [pgx - pix, pgy - piy]
			scale = self.alpha/np.max(np.abs(a_nom))
			if scale > 1:
				a_nom = scale*a_nom 

			# print()
			print('t: ', self.env.times[self.env.time_step])
			print('n_neighbor:',n_neighbor)
			print('i: ', agent_i.i)	
			# print('observation: ', observation)
			print('relative_goal: ', relative_goal)
			# print('relative_neighbors: ', relative_neighbors)
			print('sg: ', agent_i.s_g)
			# print('agent_i.p: ', agent_i.p)
			print('a_nom: ', a_nom)
			# print()
			# exit()

			if not n_neighbor == 0:

				# calculate safety controller
				# print('relative_goal: ', relative_goal)
				# print('relative_neighbors: ', relative_neighbors)

				# CVX
				a_i = cp.Variable(self.action_dim_per_agent)
				constraints = [] 

				for j in range(n_neighbor):
					rn_idx = self.state_dim_per_agent*j+np.arange(0,self.state_dim_per_agent,dtype=int)
					
					delta_p_ij = -1*relative_neighbors[rn_idx[0:2]]
					delta_v_ij = -1*relative_neighbors[rn_idx[2:]]

					A_ij = -delta_p_ij.T
					b_ij = 1/2*self.get_b_ij(delta_p_ij,delta_v_ij)
					
					constraints.append(A_ij@a_i <= b_ij) 

				constraints.append(norm_inf(a_i) <= self.alpha) 
				obj = cp.Minimize( cp.sum_squares( a_i - a_nom))
				prob = cp.Problem( obj, constraints)

				print('Solving...')

				try:
					prob.solve(verbose=True, solver = cp.GUROBI)
					# prob.solve(verbose=True)

					if prob.status in ["optimal"]:
						a_i = np.array(a_i.value)
					else:
						# do nothing 
						# a_i = 0*a_nom

						# brake
						# a_i = self.alpha*relative_goal[2:]

						# backup 
						a_i = -self.alpha*relative_goal[2:]	

				except Exception as e:
					print(e)
					# do nothing 
					# a_i = 0*a_nom
					
					# brake
					# a_i = self.alpha*relative_goal[2:]

					# backup 
					a_i = -self.alpha*relative_goal[2:]
					

				
							
			else:
				a_i = a_nom 

			# A[agent_i.i,:] = a_i + 0.1*np.random.normal(size=(1,2))
			A[agent_i.i,:] = a_i 

		# exit()
		# print('A: ',A)
		return A 

	def get_h_ij(self,dp,dv):
		h_ij = np.sqrt(4*self.alpha*(np.linalg.norm(dp) - self.Ds)) \
			+ np.matmul(dv.T, dp)/np.linalg.norm(dp)
		return h_ij

	def get_b_ij(self,dp,dv):
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



