
# Creating OpenAI gym Envs 

# standard package
from gym import Env
import numpy as np 
from collections import namedtuple

# my package
import plotter 

class Agent:
	def __init__(self,x=None,p=None,i=None,agent_memory=None,n_neighbors=None):
		self.x = x # value
		self.p = np.array(p) # position
		self.i = i # index 
		self.agent_memory = agent_memory
		self.n_neighbors = n_neighbors

		# observation history is the relative neighbor histories 
		self.observation_history = []
		for agent_j in range(n_neighbors):
			relative_neighbor_history = []
			for history_j in range(agent_memory):
				relative_neighbor_history.append(0)
			self.observation_history.append(relative_neighbor_history)


class Consensus(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None

		# get param 
		self.n_agents = param.n_agents
		self.n_neighbors = param.n_neighbors
		self.n_malicious = param.n_malicious
		self.r_comm = param.r_comm
		self.state_dim_per_agent = param.state_dim_per_agent
		self.action_dim_per_agent = param.action_dim_per_agent
		self.agent_memory = param.agent_memory

		# default parameters [SI units]
		self.n = self.state_dim_per_agent*self.n_agents
		self.m = self.action_dim_per_agent*self.n_agents

		# environment 
		self.env_state_bounds = np.array([5])

		# disturbances on initial state 
		self.init_state_disturbance = np.array([20],dtype=float)
		self.bias_disturbance = 10

		self.states_name = [
			'Node Value',
			'x-Position',
			'y-Position',
			]
		self.actions_name = [
			'Node Update'
			]

		self.param = param
		self.scale_reward = param.rl_scale_reward
		self.max_reward = 1*self.scale_reward

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
		self.update_agents_value(self.state)
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
		# return -np.abs((agent.x-self.desired_ave)/\
		# 	(self.worst_bad_node-self.desired_ave))*self.scale_reward
		return -np.abs((agent.x-self.desired_ave))*self.scale_reward
		# return (1.-np.abs((agent.x-self.desired_ave)/(self.worst_bad_node-self.desired_ave)))/self.n_agents*self.scale_reward

	def done(self):
		return False

	def observe(self, update_agents=True):

		# global observations is a list of observation_i 
		observations = []

		for agent_i in self.agents:
			p_i = agent_i.p

			# observation_i is a list of lists: first index is neighbor index, second index is history values 
			# ie observation_i is the relative neighbor histories 
			observation_i = [] 
			n_neighbor_count = -1
			for agent_j in self.agents:
				relative_neighbor_history = []
				
				if agent_j.i != agent_i.i:
					p_j = agent_j.p

					if np.linalg.norm(p_i-p_j) < self.param.r_comm:
						n_neighbor_count += 1

						# current value 
						relative_neighbor_history.append(agent_j.x-agent_i.x)

						# history values
						for i_history in range(self.agent_memory-1):
							relative_neighbor_history.append(agent_i.observation_history[n_neighbor_count][i_history])

					observation_i.append(relative_neighbor_history)
			observations.append(observation_i)

		if update_agents:
			for agent in self.agents:
				# print(agent.i)
				agent.observation_history = observations[agent.i]

		return observations	


	def unpack_observations(self,observations):

		observations_size = self.n_neighbors*self.agent_memory*self.state_dim_per_agent
		observations_array = np.empty((self.n_agents, observations_size))

		for agent in self.agents:

			observation_i_lst = [ relative_neighbor \
				for relative_neighbor_history in observations[agent.i]
				for relative_neighbor in relative_neighbor_history]

			observations_array[agent.i,:] = np.asarray(observation_i_lst)

		return observations_array

	def unpack_observation_temp_for_prev_rl_model(self,observations):

		observations_size = self.n_neighbors*self.agent_memory*self.state_dim_per_agent
		observations_array = np.empty((self.n_agents, observations_size))

		for agent in self.agents:

			observation_i_array = np.empty((observations_size))
			for i_neighbor, relative_neighbor_history in enumerate(observations[agent.i]):
				for i_history, relative_neighbor in enumerate(relative_neighbor_history):
					observation_i_array[ i_history*self.n_neighbors + i_neighbor] = relative_neighbor

			observations_array[agent.i,:] = observation_i_array
		return observations_array


	def reward(self):
		r = 0
		for agent in self.agents:
			r += self.reward_i(agent)
		return r


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


	def reset(self, initial_state=None):

		# init agents
		self.agents = []
		for i in range(self.n_agents):
			self.agents.append(Agent(i=i,agent_memory=self.agent_memory,n_neighbors=self.n_neighbors))

		if initial_state is None: 
			
			# initial behavior 
			bad_nodes = np.zeros(self.n_agents, dtype=bool)
			for _ in range(self.n_malicious):
				rand_idx = np.random.randint(0,self.n_agents)
				while bad_nodes[rand_idx]:
					rand_idx = np.random.randint(0,self.n_agents)
				bad_nodes[rand_idx] = True
			initial_behaviors = np.logical_not(bad_nodes)

			# initial positions 
			initial_positions = np.zeros((self.n_agents,2))
			d_rad = 2*np.pi/self.n_agents
			for agent in self.agents:
				initial_positions[agent.i,:] = [np.cos(d_rad*agent.i),np.sin(d_rad*agent.i)]

			# initial value
			bias = self.bias_disturbance*np.random.uniform()
			initial_values = np.zeros((self.n))
			for agent in self.agents:
				initial_values[self.agent_i_idx_to_value_i_idx(agent.i)] \
				= self.init_state_disturbance[0]*np.random.uniform() + bias

			Initial_State = namedtuple('Initial_State',['initial_values','initial_positions','initial_behaviors']) 
			initial_state = Initial_State._make((initial_values,initial_positions,initial_behaviors))

		else:

			initial_values = initial_state.initial_values
			initial_positions = initial_state.initial_positions
			initial_behaviors = initial_state.initial_behaviors


		# update agents
		self.update_agents_behavior(initial_behaviors)
		self.update_agents_position(initial_positions)
		self.update_agents_value(initial_values)

		# update environment stuff
		self.bad_nodes = np.logical_not(initial_behaviors)
		self.good_nodes = initial_behaviors
		self.state = np.copy(initial_values)
		self.desired_ave = self.good_node_average()
		self.worst_bad_node = self.find_worst_bad_node_value()
		self.time_step = 0

		return initial_state


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

	def update_agents_behavior(self, behaviors):
		for agent in self.agents:
			agent.behavior = behaviors[agent.i]

	def update_agents_value(self,values):
		for agent in self.agents:
			idx = self.agent_idx_to_state_idx(agent.i)
			agent.x = values[idx]

	def update_agents_position(self,positions):
		for agent in self.agents:
			agent.p = positions[agent.i,:]

	def agent_idx_to_state_idx(self,i):
		return i*self.state_dim_per_agent

	def agent_i_idx_to_value_i_idx(self,i):
		return i*self.state_dim_per_agent
