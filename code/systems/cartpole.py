
# Creating OpenAI gym Envs 

from gym import Env
from numpy import array,arange,diag,pi,multiply,cos,sin,dot,reshape,squeeze,vstack,mod,exp,isnan,radians,power 
from numpy.linalg import norm,pinv
from numpy.random import uniform as random_uniform
from param import param 

class CartPole(Env):

	def __init__(self):

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
			self.init_state_start = array([0,0,0,0])
			self.init_state_disturbance = array([0.25,radians(5),0.001/self.ave_dt,radians(1)/self.ave_dt])
			self.env_state_bounds = array([1.,radians(12),5/self.ave_dt,radians(180)/self.ave_dt])
		elif param.env_case is 'Swing90':
			self.init_state_start = array([0,radians(90),0,0])
			self.init_state_disturbance = array([0.1,radians(5),0,0])
			self.env_state_bounds = array([3.,radians(360),5/self.ave_dt,radians(180)/self.ave_dt])
		elif param.env_case is 'Swing180':
			self.init_state_start = array([0,radians(180),0,0])
			self.init_state_disturbance = array([0,radians(0),0,0])
			self.env_state_bounds = array([10.,radians(360),5/self.ave_dt,radians(180)/self.ave_dt])			
		else:
			print('systems.py: no case')
			exit()

		self.W = diag([0.01,1,0,0])
		self.max_error = 2*self.env_state_bounds
		self.max_penalty = dot(self.max_error.T,dot(self.W,self.max_error))

		self.states_name = [
			'Cart Position [m]',
			'Pole Angle [rad]',
			'Cart Velocity [m/s]',
			'Pole Velocity [rad/s]']
		self.actions_name = [
			'Cart Acceleration [m/s^2]']

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
		state_ref = param.ref_trajectory[:,self.time_step]
		error = self.state - state_ref
		C = 1.
		# r = exp(-C*dot(error.T,dot(W,error)))
		return 1 - power(dot(error.T,dot(self.W,error))/self.max_penalty,1/6)
		# return 1 - power(dot(error.T,dot(self.W,error))/self.max_penalty,1)

	def max_reward(self):
		return 1.
		
	def reset(self, initial_state = None):
		if initial_state is None:
			self.state = self.init_state_start+multiply(self.init_state_disturbance, random_uniform(size=(4,)))
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
		dt = self.times[self.time_step+1]-self.times[self.time_step]

		# s = [q,qdot], q = [x,th]
		a = reshape(a,(len(a),1))
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
		# if sp1[1] > pi:
		# 	sp1[1] -= 2*pi
		# if sp1[1] < -pi:
		# 	sp1[1] += 2*pi
		return sp1

	def env_barrier(self,action):
		pass