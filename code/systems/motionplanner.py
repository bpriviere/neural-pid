
# Creating OpenAI gym Envs 

# standard package
from gym import Env
import numpy as np 

# my package
import plotter 

class MotionPlanner(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None
		self.ave_dt = np.mean(self.times[1:]-self.times[:-1])

		self.n_agents = 1
		self.n_obs = 1

		# 
		self.delta_a = 0.2
		self.delta_o = 2
		self.delta_g = 0.2
		self.o_x = 0
		self.o_y = 0
		self.s_g = np.array([4,0,0,0])

		# default parameters [SI units]
		self.n = 4
		self.m = 2

		# environment 
		self.env_state_bounds = np.array(
			[5,5,5/self.ave_dt,5/self.ave_dt])
		self.init_state_start = np.array(
			[0,0,0,0])
		self.init_state_disturbance = np.array(
			[self.env_state_bounds[0],self.env_state_bounds[1],0,0])

		self.states_name = [
			'x-Position [m]',
			'y-Position [m]',
			'x-Velocity [m/s]',
			'y-Velocity [m/s]',]
		self.actions_name = [
			'x-Acceleration [m/s^2]',
			'y-Acceleration [m/s^2]']

		# weigh error
		self.W = np.diag([1,1,0,0])
		self.max_e = self.s_g + self.env_state_bounds
		self.max_d = np.dot(np.dot(self.max_e,self.W),self.max_e)
		self.max_reward = 1.


	def render(self):
		fig,ax = plotter.make_fig(axlim = [self.env_state_bounds[0],self.env_state_bounds[1]])
		title = 'State at t: ' + str(np.round(self.times[self.time_step]))

		for j in range(self.n_obs):
			plotter.plot_circle(self.o_x,self.o_y,self.delta_o,fig=fig,ax=ax,label='obs')
		for i in range(self.n_agents):
			plotter.plot_circle(self.s[0],self.s[1],self.delta_a,fig=fig,ax=ax,label='agent')
		plotter.plot_circle(self.s_g[0],self.s_g[1],self.delta_g,fig=fig,ax=ax,title=title,label='goal')
		

	def step(self, a):
		self.s = self.f(self.s, a)
		d = self.done() 
		r = self.reward()
		self.time_step += 1
		return self.s, r, d, {}


	def done(self):
		return self.collision_check_obs() or self.collision_check_env()


	def reward(self):
		if self.collision_check_obs() or self.collision_check_env():
			return -100
		else:
			e = self.s - self.s_g
			d = np.dot(np.dot(e,self.W),e)
			return (1 - d/self.max_d)**2.


	def reset(self, initial_state = None):
		
		if initial_state is None:
			self.s = self.init_state_start+np.multiply(
					self.init_state_disturbance,np.random.uniform(size=(4,)))
			# self.s[0] = np.min((self.s[0],0))
			# check that you don't initialize inside obstacle! 
			while self.collision_check_obs():
				self.s = self.init_state_start+np.multiply(
					self.init_state_disturbance,np.random.uniform(size=(4,)))
				# self.s[0] = np.min((self.s[0],0))
		else:
			self.s = initial_state
			# check that you don't initialize inside obstacle! 
			if self.collision_check_obs():
				print('Initialized Inside Obstacle')
				exit()
		self.time_step = 0
		return np.array(self.s) 


	def f(self,s,a):
		dt = self.times[self.time_step+1] - self.times[self.time_step]
		sdot = np.array((
			self.s[2],self.s[3],a[0],a[1]))
		s = s + sdot*dt
		return np.array(s)


	def collision_check_obs(self):
		if np.linalg.norm(
			self.s[0:2]-[self.o_x,self.o_y]) <= self.delta_o+self.delta_a:
			return True
		else:
			return False


	def collision_check_goal(self):
		if np.linalg.norm(
			self.s[0:2]-self.s_g[0:2]) <= self.delta_g+self.delta_a:
			return True
		return False


	def collision_check_env(self):
		if np.abs(self.s[0]) + self.delta_a > self.env_state_bounds[0]:
			return True
		elif np.abs(self.s[1]) + self.delta_a > self.env_state_bounds[1]:
			return True
		return False 

