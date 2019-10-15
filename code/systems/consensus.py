
# Creating OpenAI gym Envs 

# standard package
from gym import Env
import numpy as np 

# my package
import plotter 

class Agent:
	def __init__(self,v=None,p=None,i=None):
		self.v = v # value
		self.p = np.array(p) # position
		self.i = i # index 

class Consensus(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None

		self.n_agents = 3

		# default parameters [SI units]
		self.n = 1*self.n_agents
		self.m = 1*self.n_agents

		self.agents = []
		p = [[0,0],[1,0],[0,1]]
		for i in range(self.n_agents):
			agent = Agent(p=p[i],i=i)
			self.agents.append(agent)

		# determine malicious node 
		self.s1_idx = np.ones(self.n_agents, dtype=bool)
		self.s1_idx[np.random.randint(0,self.n_agents)] = False
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
		pass

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
				if np.linalg.norm(p_i-p_j) < self.param.r_comm:
					observation_i.append(agent_j.v-agent_i.v)
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

		sp1 = np.zeros((self.n))
		for i in range(self.n_agents):
			if self.s1_idx[i]:
				sp1[i] = self.s[i] + a[i]
			else:
				sp1[i] = self.s[i]

		self.update_agents(sp1)
		return sp1

	def update_agents(self,s):
		for agent_i in self.agents:
			agent_i.v = s[agent_i.i]