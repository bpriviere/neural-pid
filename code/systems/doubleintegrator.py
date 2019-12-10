
# Creating OpenAI gym Envs 

# standard package
from gym import Env
from collections import namedtuple
import numpy as np 
import torch 

# my package
import plotter 

class Agent:
	def __init__(self,i):
		self.i = i 
		self.s = None
		self.p = None
		self.v = None
		self.s_g = None

class DoubleIntegrator(Env):

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
		self.v_min = param.v_min
		self.v_max = param.v_max

		# default parameters [SI units]
		self.n = self.state_dim_per_agent*self.n_agents
		self.m = self.action_dim_per_agent*self.n_agents

		# init agents
		self.agents = []
		for i in range(self.n_agents):
			self.agents.append(Agent(i))

		# environment 
		self.init_state_mean = 0.
		self.init_state_var = 10.

		self.states_name = [
			'x-Position [m]',
			'y-Position [m]',
			'x-Velocity [m/s]',
			'y-Velocity [m/s]',
			]
		self.actions_name = [
			'x-Acceleration [m/s^2]',
			'y-Acceleration [m/s^2]'
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
			relative_goal = torch.Tensor(agent_i.s_g - s_i)
			relative_neighbors = []
			for agent_j in self.agents:
				if agent_j.i != agent_i.i:
					p_j = agent_j.p
					if np.linalg.norm(p_i-p_j) < self.param.r_comm:
						s_j = agent_j.s
						relative_neighbors.append(torch.Tensor(s_j-s_i))
			observation_i = Observation._make((relative_goal,relative_neighbors))

			# convert to new format
			obs_array = np.zeros(self.state_dim_per_agent*(1+len(observation_i.relative_neighbors)))
			obs_array[0:self.state_dim_per_agent] = observation_i.relative_goal
			for i in range(len(observation_i.relative_neighbors)):
				idx = (i+1)*self.state_dim_per_agent + np.arange(0,self.state_dim_per_agent,dtype=int)
				obs_array[idx ] = observation_i.relative_neighbors[i]

			observations.append(obs_array)
			# observations.append(observation_i)
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


	def reset(self, initial_state=None):
		self.time_step = 0				
		if initial_state is None:

			initial_state = np.zeros((self.n))
			for agent_i in self.agents:
				agent_i.s = self.find_collision_free_state('initial')
				# agent_i.s_g = self.find_collision_free_state('goal')
				agent_i.s_g = -agent_i.s
				idx = self.agent_idx_to_state_idx(agent_i.i) + \
					np.arange(0,self.state_dim_per_agent)
				initial_state[idx] = agent_i.s
			self.s = initial_state
		else:
			# print(initial_state)
			self.s = initial_state.start

			# assign goal state 
			for agent in self.agents:
				idx = self.agent_idx_to_state_idx(agent.i) + \
					np.arange(0,self.state_dim_per_agent)
				# print(idx)
				agent.s_g = initial_state.goal[idx]

		self.update_agents(self.s)
		return np.copy(self.s)


	def find_collision_free_state(self,config):
		collision = True
		count = 0 
		while collision:
			count += 1
			collision = False
			s = self.init_state_mean + \
					self.init_state_var*np.random.uniform(size=(self.state_dim_per_agent))
			for agent_j in self.agents:
				if agent_j.s is not None and agent_j.s_g is not None:
					if config == 'initial': 
						dist = np.linalg.norm(agent_j.s[0:2] - s[0:2])
					elif config == 'goal':
						dist = np.linalg.norm(agent_j.s_g[0:2] - s[0:2])
					if dist < 2*self.r_agent:
						collision = True
						break

			if count > 1000:
				print('Infeasible initial conditions')
				exit()

		return s 


	def next_state(self,s,a):

		sp1 = np.zeros((self.n))
		dt = self.times[self.time_step+1]-self.times[self.time_step]

		# single integrator
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			p_idx = np.arange(idx,idx+2)
			v_idx = np.arange(idx+2,idx+4)
			sp1[p_idx] = self.s[p_idx] + self.s[v_idx]*dt
			sp1[v_idx] = self.s[v_idx] + a[agent_i.i,:]*dt 

			sp1[v_idx] = np.clip(sp1[v_idx],self.v_min,self.v_max)

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