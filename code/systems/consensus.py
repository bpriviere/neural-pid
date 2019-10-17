
# Creating OpenAI gym Envs 

# standard package
from gym import Env
import numpy as np 

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
		self.r_comm = param.r_comm
		self.state_dim_per_agent = 3
		self.action_dim_per_agent = 1

		# default parameters [SI units]
		self.n = self.state_dim_per_agent*self.n_agents
		self.m = self.action_dim_per_agent*self.n_agents

		# initialize agents
		self.agents = []
		for i in range(self.n_agents):
			self.agents.append(Agent(i=i))

		# determine malicious node 
		self.s2_idx = np.array((1,0,0,0,0))
		self.s1_idx = np.ones(self.n_agents, dtype=bool) - self.s2_idx
		self.true_ave = None

		# environment 
		self.env_state_bounds = np.array([5])
		self.init_state_start = np.zeros((self.n_agents))
		self.init_state_disturbance = 10*np.ones((self.n_agents))

		self.states_name = [
			'Node Value'
			]
		self.actions_name = [
			'Node Update'
			]

		self.param = param


	def render(self):

		fig,ax = plotter.make_fig()
		# plot edges
		for agent_i in self.agents:
			p_i = agent_i.p 
			for agent_j in self.agents:
				p_j = agent_j.p 

				if np.linalg.norm(p_i - p_j) < self.r_comm:
					plotter.plot([p_i[0],p_j[0]],[p_i[1],p_j[1]],fig=fig,ax=ax)

		# plot nodes
		for agent_i in self.agents:
			p_i = agent_i.p 
			plotter.plot_circle(p_i[0],p_i[1],0.25,fig=fig,ax=ax)

	def step(self, a):
		self.s = self.next_state(self.s, a)
		d = self.done()
		r = self.reward()
		self.time_step += 1
		return self.s, r, d, {}

	def done(self):
		return False

	def observe(self):
		observation = []
		for agent_i in self.agents:
			p_i = agent_i.p
			observation_i = [] 
			for agent_j in self.agents:
				p_j = agent_j.p
				if np.linalg.norm(p_i-p_j) < self.r_comm:
					observation_i.append(agent_j.x-agent_i.x)
			observation.append(observation_i)
		return observation

	def reward(self):
		if self.true_ave is None:
			self.true_ave = np.mean(self.s[self.s1_idx])
		curr_ave = np.mean(self.s[self.s1_idx])
		return 1-np.linalg.norm(self.s[self.s1_idx]-self.true_ave)/\
			np.linalg.norm(self.s[~self.s1_idx]-self.true_ave)


	def reset(self, initial_state = None):
		self.time_step = 0				
		if initial_state is None:
			self.s = self.init_state_start+np.multiply(
					self.init_state_disturbance,np.random.uniform(size=(self.n,)))
		else:
			self.s = initial_state

		self.update_agents(self.s)			
		return np.array(self.s)


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

			if self.s1_idx[agent_i.i]:
				sp1[idx] = self.s[idx] + a[agent_i.i]*dt
			else:
				sp1[idx] = self.s[idx]

		self.update_agents(sp1)
		return sp1

	def update_agents(self,s):
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			agent_i.x = s[idx+0]
			agent_i.p = s[idx+1:idx+3]

	def agent_idx_to_state_idx(self,i):
		return i*self.state_dim_per_agent