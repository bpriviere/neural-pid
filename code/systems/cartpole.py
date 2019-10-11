
# Creating OpenAI gym Envs 

from gym import Env
import numpy as np

class CartPole(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None
		self.ave_dt = self.times[1]-self.times[0]

		# default parameters [SI units]
		self.n = 4
		self.m = 1
		self.mass_cart = 1.0
		self.mass_pole = 0.1
		self.length_pole = 0.5
		self.g = 9.81
		if param.env_case is 'SmallAngle':
			self.init_state_start = np.array([0,0,0,0])
			self.init_state_disturbance = np.array([0.25,np.radians(5),0.001/self.ave_dt,np.radians(1)/self.ave_dt])
			self.env_state_bounds = np.array([1.,np.radians(12),5/self.ave_dt,np.radians(180)/self.ave_dt])
		elif param.env_case is 'Swing90':
			self.init_state_start = np.array([0,np.radians(90),0,0])
			self.init_state_disturbance = array([0.1,np.radians(5),0,0])
			self.env_state_bounds = np.array([3.,np.radians(360),5/self.ave_dt,np.radians(180)/self.ave_dt])
		elif param.env_case is 'Swing180':
			self.init_state_start = np.array([0,np.radians(180),0,0])
			self.init_state_disturbance = np.array([0,np.radians(0),0,0])
			self.env_state_bounds = np.array([10.,np.radians(360),5/self.ave_dt,np.radians(180)/self.ave_dt])
		else:
			raise Exception('param.env_case invalid ' + param.env_case)

		self.W = np.diag([0.01,1,0,0])
		self.max_error = 2*self.env_state_bounds
		self.max_penalty = np.dot(self.max_error.T,np.dot(self.W,self.max_error))
		self.max_reward = 1.

		self.states_name = [
			'Cart Position [m]',
			'Pole Angle [rad]',
			'Cart Velocity [m/s]',
			'Pole Velocity [rad/s]']
		self.actions_name = [
			'Cart Acceleration [m/s^2]']
		self.param = param

	def step(self, action):
		state = self.state
		initial_time = self.times[self.time_step]
		final_time = self.times[self.time_step + 1]
		self.state = self.f(state, action)
		done = abs(state[0]) > self.env_state_bounds[0] \
			or abs(state[1]) > self.env_state_bounds[1] \
			or self.time_step == len(self.times)-1
		r = self.reward()
		self.time_step += 1
		return self.state, r, done, {}

	def reward(self):
		state_ref = self.param.ref_trajectory[:,self.time_step]
		error = self.state - state_ref
		C = 1.
		# r = exp(-C*dot(error.T,dot(W,error)))
		return 1 - np.power(np.dot(error.T,np.dot(self.W,error))/self.max_penalty,1/6)
		# return 1 - power(dot(error.T,dot(self.W,error))/self.max_penalty,1)

		
	def reset(self, initial_state = None):
		if initial_state is None:
			self.state = self.init_state_start+np.multiply(self.init_state_disturbance, np.random.uniform(size=(4,)))
		else:
			self.state = initial_state
		self.time_step = 0
		return np.array(self.state)

	def f(self,s,a):
		# input:
		# 	s, nd array, (n,)
		# 	a, nd array, (m,1)
		# output
		# 	sp1, nd array, (n,)

		# parameters
		m_p = self.mass_pole
		m_c = self.mass_cart
		l = self.length_pole
		g = self.g
		dt = self.times[self.time_step+1]-self.times[self.time_step]

		# s = [q,qdot], q = [x,th]
		a = np.reshape(a,(self.m,1))
		q = np.reshape(s[0:2],(2,1))
		qdot = np.reshape(s[2:],(2,1))
		th = s[1]
		thdot = s[3]

		# EOM from learning+control@caltech
		D = np.array([[m_c+m_p,m_p*l*np.cos(th)],[m_p*l*np.cos(th),m_p*(l**2)]])
		C = np.array([[0,-m_p*l*thdot*np.sin(th)],[0,0]])
		G = np.array([[0],[-m_p*g*l*np.sin(th)]])
		B = np.array([[1],[0]])
		qdotdot = np.dot(np.linalg.pinv(D), np.dot(B,a) - np.dot(C,qdot) - G)

		# euler integration
		sp1 = np.squeeze(np.reshape(s,(len(s),1)) + np.vstack([qdot, qdotdot]) * dt)
		# if sp1[1] > pi:
		# 	sp1[1] -= 2*pi
		# if sp1[1] < -pi:
		# 	sp1[1] += 2*pi

		return sp1

	def env_barrier(self,action):
		pass