
# Creating OpenAI gym Envs 

# standard package
from gym import Env
from collections import namedtuple
import numpy as np 
import torch 

# my package
import plotter 

class Agent:
	def __init__(self,s=None,i=None):
		self.s = s
		self.i = i # index 
		self.p = None
		self.v = None

class SingleIntegrator(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None

		self.n_agents = param.n_agents
		self.state_dim_per_agent = 4
		self.action_dim_per_agent = 2
		self.r_agent = param.r_agent

		# control lim
		self.a_min = param.a_min
		self.a_max = param.a_max

		# default parameters [SI units]
		self.n = self.state_dim_per_agent*self.n_agents
		self.m = self.action_dim_per_agent*self.n_agents

		self.agents = []
		for i in range(self.n_agents):
			self.agents.append(Agent(i=i))

		# environment 
		self.env_state_bounds = np.array([5])
		self.init_state_start = np.zeros((self.n_agents))
		self.init_state_disturbance = 10*np.ones((self.n_agents))

		self.states_name = [
			'x-Position [m]',
			'y-Position [m]',
			'x-Velocity [m/s]',
			'y-Velocity [m/s]',
			]
		self.actions_name = [
			'Node Update'
			]

		self.param = param
		self.max_reward = 0 


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
		Observation = namedtuple('Observation',['relative_goal','relative_neighbors']) 

		observations = []
		for agent_i in self.agents:
			p_i = agent_i.p
			s_i = agent_i.s
			relative_goal = agent_i.s_g - s_i
			relative_neighbors = []
			for agent_j in self.agents:
				if agent_j.i != agent_i.i:
					p_j = agent_j.p
					if np.linalg.norm(p_i-p_j) < self.param.r_comm:
						s_j = agent_j.s
						relative_neighbors.append(s_j-s_i)
			observation_i = Observation._make((relative_goal,relative_neighbors))
			observations.append(observation_i)
		return observations

	def reward(self):
		minDist = np.Inf
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			pos_i = self.s[idx:idx+2]
			for agent_j in self.agents:
				if agent_i != agent_j:
					idx = self.agent_idx_to_state_idx(agent_j.i)
					pos_j = self.s[idx:idx+2]
					dist = np.linalg.norm(pos_i - pos_j)
					if dist < minDist:
						minDist = dist
		if minDist < 2*self.r_agent:
			return -1
		return 0


	def reset(self, initial_state = None):
		self.time_step = 0				
		if initial_state is None:
			self.s = self.init_state_start+np.multiply(
					self.init_state_disturbance,np.random.uniform(size=(self.n,)))
		else:
			self.s = initial_state

		# assign goal state 
		for agent in self.agents:
			idx = self.agent_idx_to_state_idx(agent.i) + \
				np.arange(0,self.state_dim_per_agent)
			agent.s_g = -initial_state[idx]

		self.update_agents(self.s)			
		return np.array(self.s) 


	def next_state(self,s,a):

		sp1 = np.zeros((self.n))
		dt = self.times[self.time_step+1]-self.times[self.time_step]

		# single integrator
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			p_idx = np.arange(idx,idx+2)
			v_idx = np.arange(idx+2,idx+4)
			sp1[p_idx] = self.s[p_idx] + self.s[v_idx]*dt
			sp1[v_idx] = a[agent_i.i,:]
		self.update_agents(sp1)
		return sp1
		
	def next_state_training_state_loss(self,s,a):
		# input: ONE agent state, and ONE agent action
		# output: increment of state
		# used in train_il for state-loss function 

		s = torch.from_numpy(s[0:self.state_dim_per_agent]).float()

		dt = self.times[1]-self.times[0]
		I = torch.eye((self.state_dim_per_agent))
		A = torch.from_numpy(np.array((
			[[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]))).float()
		B = torch.from_numpy(np.array((
			[[0,0],[0,0],[1,0],[0,1]]))).float()
		sp1 = (I + A*dt)@s + B@a
		return sp1		

	def update_agents(self,s):
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			agent_i.p = s[idx:idx+2]
			agent_i.v = s[idx+2:idx+4]
			agent_i.s = np.concatenate((agent_i.p,agent_i.v))

	def agent_idx_to_state_idx(self,i):
		return self.state_dim_per_agent*i 

	def visualize(self,states,dt):

		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		import time 

		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		for i in range(self.n_agents):
			vis["agent"+str(i)].set_object(g.Sphere(self.r_agent))

		while True:
			for state in states:
				for i in range(self.n_agents):
					idx = self.agent_idx_to_state_idx(i) + np.arange(0,2)
					pos = state[idx]
					vis["agent" + str(i)].set_transform(tf.translation_matrix([pos[0], pos[1], 0]))
				time.sleep(dt)