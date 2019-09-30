
# Creating OpenAI gym Envs 

from gym import Env
from numpy import array,arange,diag,pi,multiply,cos,sin,dot,reshape,squeeze,vstack
from numpy.linalg import norm,pinv
from numpy.random import uniform as random_uniform
from param import param 

class CartPole(Env):

	def __init__(self):

		self.times = param.get('times')
		self.dt = self.times[1] - self.times[0]
		self.state = None
		self.time_step = None

		# default [SI units]
		self.n = param.get('sys_n')
		self.m = param.get('sys_m')
		self.actions = param.get('sys_actions')
		self.card_A = param.get('sys_card_A')
		self.mass_cart = param.get('sys_mass_cart') 
		self.mass_pole = param.get('sys_mass_pole')
		self.length_pole = param.get('sys_length_pole')
		self.init_state_bounds = param.get('sys_init_state_bounds')
		self.objective = param.get('sys_objective')
		self.pos_bounds = param.get('sys_pos_bounds')
		self.angle_bounds_deg = param.get('sys_angle_bounds_deg')
		self.g = param.get('sys_g')

	def step(self, action):
		state = self.state
		initial_time = self.times[self.time_step]
		final_time = self.times[self.time_step + 1]
		self.state = self.f(state, action)
		done = abs(state[0]) > self.pos_bounds \
			or abs(state[1]) > self.angle_bounds_deg*pi/180. \
			or self.time_step == len(self.times)-1
		# done = False
		r = self.reward()
		self.time_step += 1
		return self.state, r, done, {}

	def reward(self):
		if self.objective is 'stabilize':
			return 0.01
		elif self.objective is 'track':
			state_ref = param.get('reference_trajectory')[:,self.time_step]
			error = self.state - state_ref
			W = diag([0.0004,0.0625,0,0])
			return 0.01 - dot(error.T,dot(W,error))

	def max_reward(self):
		return 0.01

	def reset(self, initial_state = None):
		if initial_state is None:
			self.state = multiply(self.init_state_bounds, random_uniform(size=(4,)))
		else:
			self.state = initial_state
		self.time_step = 0
		return array(self.state)

	def f(self,s,a):

		# parameters
		m_p = self.mass_pole
		m_c = self.mass_cart
		l = self.length_pole
		g = self.g
		dt = self.dt

		# s = [q,qdot], q = [x,th]
		q = reshape(s[0:2],(2,1))
		qdot = reshape(s[2:],(2,1))
		th = s[1]
		thdot = s[3]

		# EOM from learning+control@caltech
		D = array([[m_c+m_p,m_p*l*cos(th)],[m_p*l*cos(th),m_p*(l**2)]])
		C = array([[0,-m_p*l*thdot*sin(th)],[0,0]])
		G = array([[0],[-m_p*g*l*sin(th)]])
		B = array([[1],[0]])
		qdotdot = dot(pinv(D), dot(B,a) - dot(C,qdot) - G)

		# euler integration
		sp1 = squeeze(reshape(s,(len(s),1)) + vstack([qdot, qdotdot]) * dt)
		return sp1