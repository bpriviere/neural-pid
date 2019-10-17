
# Creating OpenAI gym Envs 

# standard package
from gym import Env
import numpy as np 

# my package
import plotter 

class Agent:
	def __init__(self,s=None,i=None):
		self.s = s
		self.i = i # index 

class SingleIntegrator(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None

		self.n_agents = param.n_agents
		self.config_dim = 4
		self.agent_radius = 0.75

		# default parameters [SI units]
		self.n = self.config_dim*self.n_agents
		self.m = 2*self.n_agents

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
			o = np.concatenate((agent_i.p, agent_i.v, agent_i.sg))
			for agent_j in self.agents:
				p_j = agent_j.p
				if np.linalg.norm(p_i-p_j) < self.param.r_comm:
					o = np.concatenate((o,agent_j.p-agent_i.p, agent_j.v-agent_i.v))
			observation_i.append(o)
			observation.append(observation_i)
		return observation


	def reward(self):
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
				np.arange(0,self.config_dim)
			agent.sg = -initial_state[idx]

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

	def update_agents(self,s):
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			agent_i.p = s[idx:idx+2]
			agent_i.v = s[idx+2:idx+4]

	def agent_idx_to_state_idx(self,i):
		return self.config_dim*i 		

	def visualize(self,states,dt):

		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		import time 

		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		for i in range(self.n_agents):
			vis["agent"+str(i)].set_object(g.Sphere(self.agent_radius))

		while True:
			for state in states:
				for i in range(self.n_agents):
					idx = self.agent_idx_to_state_idx(i) + np.arange(0,2)
					pos = state[idx]
					vis["agent" + str(i)].set_transform(tf.translation_matrix([pos[0], pos[1], 0]))
				time.sleep(dt)