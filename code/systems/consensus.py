
# Creating OpenAI gym Envs 

# standard package
from gym import Env
import numpy as np 
from collections import namedtuple

# my package
import plotter 

class Agent:
	def __init__(self,x=None,p=None,i=None):
		self.x = x # value
		self.p = np.array(p) # position
		self.i = i # index 

class Consensus(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None

		self.n_agents = param.n_agents
		self.n_malicious = param.n_malicious
		self.r_comm = param.r_comm
		self.state_dim_per_agent = param.state_dim_per_agent
		self.action_dim_per_agent = param.action_dim_per_agent

		# default parameters [SI units]
		self.n = self.state_dim_per_agent*self.n_agents
		self.m = self.action_dim_per_agent*self.n_agents

		# initialize agents
		self.agents = []
		for i in range(self.n_agents):
			self.agents.append(Agent(i=i))

		# determine malicious nodes 
		self.bad_nodes = np.zeros(self.n_agents, dtype=bool)
		for _ in range(self.n_malicious):
			# rand_idx = np.random.randint(0,self.n_agents)
			# while self.bad_nodes[rand_idx]:
			# 	rand_idx = np.random.randint(0,self.n_agents)
			# self.bad_nodes[rand_idx] = True
			self.bad_nodes[1] = True

		self.good_nodes = np.logical_not(self.bad_nodes)
		self.desired_ave = None

		print('Good Nodes:', self.good_nodes)
		print('Bad Nodes:', self.bad_nodes)

		# environment 
		self.env_state_bounds = np.array([5])

		# initialize 
		
		# self.init_state_start = np.array([
		# 	[0,0,0],
		# 	[0,1,0],
		# 	[0,0,1],
		# 	[0,1,1],
		# 	[0,2,1]],dtype=float)

		init_positions = np.zeros((self.n_agents,2))
		d_rad = 2*np.pi/self.n_agents
		for agent in self.agents:
			init_positions[agent.i,:] = [np.cos(d_rad*agent.i),np.sin(d_rad*agent.i)]
		self.update_agents_position(init_positions)

		# disturbances on initial state 
		self.init_state_disturbance = np.array(
			[10],dtype=float)

		self.states_name = [
			'Node Value',
			'x-Position',
			'y-Position',
			]
		self.actions_name = [
			'Node Update'
			]

		self.param = param
		self.max_reward = 1 


	def render(self):

		fig,ax = plotter.make_fig()
		ax.set_aspect('equal')
		# plot edges
		for agent_i in self.agents:
			color = 'black'
			p_i = agent_i.p 
			for agent_j in self.agents:
				p_j = agent_j.p 
				if np.linalg.norm(p_i - p_j) < self.r_comm:
					plotter.plot([p_i[0],p_j[0]],[p_i[1],p_j[1]],fig=fig,ax=ax,color=color)

		# plot nodes
		for agent_i in self.agents:
			p_i = agent_i.p 
			if self.good_nodes[agent_i.i]:
				color = 'blue'
			else: 
				color = 'red'
			plotter.plot_circle(p_i[0],p_i[1],0.1,fig=fig,ax=ax,color=color)

		# 
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()

		buff = 0.2
		ax.set_xlim(xlim[0]-buff,xlim[1]+buff)
		ax.set_ylim(ylim[0]-buff,ylim[1]+buff)
		ax.set_axis_off()


	def step(self, a):
		self.state = self.next_state(self.state, a)
		self.update_agents()
		d = self.done()
		r = self.reward()
		self.time_step += 1
		return self.state, r, d, {}

	def step_i(self, agent_i, a_i):
		# step function for a single agent, used for training
		s_i = self.next_state_i(agent_i,a_i)
		r_i = self.reward_i(agent_i)
		d_i = self.done()
		return s_i,r_i,d_i

	def next_state_i(self,agent,a):
		return agent.x + a 

	def reward_i(self,agent):
		return -np.abs(agent.x-self.desired_ave)

	def done(self):
		return False


	def observe(self):
		observations = []
		for agent_i in self.agents:
			p_i = agent_i.p
			relative_neighbors = []
			for agent_j in self.agents:
				if agent_j.i != agent_i.i:
					p_j = agent_j.p
					if np.linalg.norm(p_i-p_j) < self.param.r_comm:
						relative_neighbors.append(agent_j.x-agent_i.x)
			observation_i = relative_neighbors
			observations.append(observation_i)
		return observations


	def reward(self):
		curr_vals = self.state[np.arange(0,self.n_agents,self.state_dim_per_agent)]
		return 1-np.linalg.norm(curr_vals-self.desired_ave)/\
			np.linalg.norm(self.worst_bad_node-self.desired_ave)


	def good_node_average(self):
		summ = 0
		count = 0 
		for agent_i in self.agents:
			if self.good_nodes[agent_i.i]:
				count += 1
				summ += agent_i.x 
		return summ/count


	def find_worst_bad_node_value(self):
		maxx = 0 
		for agent_i in self.agents:
			if self.bad_nodes[agent_i.i] and \
			maxx <= abs(agent_i.x - self.desired_ave):
				maxx = abs(agent_i.x)
		return maxx


	def reset(self, initial_state = None):
		self.time_step = 0				
		if initial_state is None:
			self.state = np.zeros((self.state_dim))
			for agent in self.agents:
				self.state[self.agent_i_idx_to_value_i_idx(agent_i.i)] 
				= self.init_state_disturbance[0]*np.random.uniform() 
		else:
			self.state = initial_state

		for agent_i in self.agents:
			if self.bad_nodes[agent_i.i]:
				self.state[
				self.agent_i_idx_to_value_i_idx(agent_i.i)] = 10 
		

		self.update_agents()			
		self.desired_ave = self.good_node_average()
		self.worst_bad_node = self.find_worst_bad_node_value()
		return np.array(self.state)


	def next_state(self,s,a):

		# input:
		# 	s, ndarray, (n,) 
		# 	a, ndarray, (m,1)
		# output:
		# 	sp1, ndarray, (n,)

		sp1 = s
		dt = self.times[self.time_step+1]-self.times[self.time_step]		
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			if self.good_nodes[agent_i.i]: 
				sp1[idx] = s[idx] + a[agent_i.i]
			else:
				sp1[idx] = s[idx]
		return sp1

	def update_agents(self):
		s = self.state
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			agent_i.x = s[idx]

	def update_agents_position(self,p):
		for agent in self.agents:
			agent.p = p[agent.i,:]			

	def agent_idx_to_state_idx(self,i):
		return i*self.state_dim_per_agent

	def agent_i_idx_to_value_i_idx(self,i):
		return i*self.state_dim_per_agent
