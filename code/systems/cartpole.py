
# Creating OpenAI gym Envs 

from gym import Env
import numpy as np
import autograd.numpy as agnp  # Thinly-wrapped numpy

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
		self.a_min = np.array([-10])
		self.a_max = np.array([10])
		if param.env_case is 'SmallAngle':
			self.init_state_start = np.array([0,0,0,0])
			self.init_state_disturbance = np.array([0.25,np.radians(5),0.001/self.ave_dt,np.radians(1)/self.ave_dt])
			self.env_state_bounds = np.array([1.,np.radians(12),5/self.ave_dt,np.radians(180)/self.ave_dt])
		elif param.env_case is 'Swing90':
			self.init_state_start = np.array([0,np.radians(90),0,0])
			self.init_state_disturbance = np.array([0.1,np.radians(5),0,0])
			self.env_state_bounds = np.array([3.,np.radians(360),5/self.ave_dt,np.radians(180)/self.ave_dt])
		elif param.env_case is 'Swing180':
			self.init_state_start = np.array([0,np.radians(180),0,0])
			self.init_state_disturbance = np.array([0,np.radians(0),0,0])
			self.env_state_bounds = np.array([10.,np.radians(360),5/self.ave_dt,np.radians(180)/self.ave_dt])
		elif param.env_case is 'Any90':
			self.init_state_start = np.array([0,np.radians(0),0,0])
			self.init_state_disturbance = np.array([3.,np.radians(90),4.,10.])
			self.env_state_bounds = np.array([3.,np.radians(360),4., 10.])
		else:
			raise Exception('param.env_case invalid ' + param.env_case)

		self.s_min = -self.env_state_bounds
		self.s_max = -self.s_min

		self.W = np.diag([0.01,1,0,0])
		self.max_error = 2*self.env_state_bounds
		self.max_penalty = np.dot(self.max_error.T,np.dot(self.W,self.max_error))
		self.max_reward = 1.

		self.states_name = [
			'Cart Position [m]',
			'Pole Angle [rad]',
			'Cart Velocity [m/s]',
			'Pole Velocity [rad/s]']
		self.deduced_state_names = []
		self.actions_name = [
			'Cart Acceleration [m/s^2]']
		self.param = param

	def step(self, action):
		# state = self.state
		# initial_time = self.times[self.time_step]
		# final_time = self.times[self.time_step + 1]
		self.state = self.next_state(self.state, action)
		done = abs(self.state[0]) > self.env_state_bounds[0] \
			or abs(self.state[1]) > self.env_state_bounds[1] \
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

	# xdot = f(x, u)
	def f(self, x, u):
		# input:
		# 	x, nd array, (n,)
		# 	u, nd array, (m,1)
		# output
		# 	sp1, nd array, (n,)

		# parameters
		m_p = self.mass_pole
		m_c = self.mass_cart
		l = self.length_pole
		g = self.g

		# s = [q,qdot], q = [x,th]
		u = np.reshape(u,(self.m,1))
		q = np.reshape(x[0:2],(2,1))
		qdot = np.reshape(x[2:],(2,1))
		th = x[1]
		thdot = x[3]

		# EOM from learning+control@caltech
		# D = np.array([[m_c+m_p,m_p*l*np.cos(th)],[m_p*l*np.cos(th),m_p*(l**2)]])
		a = m_c+m_p
		b = m_p*l*np.cos(th)
		c = m_p*l*np.cos(th)
		d = m_p*(l**2)
		Dinv = 1/(a*d-b*c) * np.array([[d, -b], [-c, a]])

		C = np.array([[0,-m_p*l*thdot*np.sin(th)],[0,0]])
		G = np.array([[0],[-m_p*g*l*np.sin(th)]])
		B = np.array([[1],[0]])
		qdotdot = np.dot(Dinv, np.dot(B,u) - np.dot(C,qdot) - G)

		res = np.vstack([qdot, qdotdot])
		return res

	# xdot = f(x, u)
	# use autograd here so we can support SCP
	def f_scp(self, x, u):
		# input:
		# 	x, nd array, (n,)
		# 	u, nd array, (m,1)
		# output
		# 	sp1, nd array, (n,)

		# parameters
		m_p = self.mass_pole
		m_c = self.mass_cart
		l = self.length_pole
		g = self.g

		# s = [q,qdot], q = [x,th]
		u = agnp.reshape(u,(self.m,1))
		q = agnp.reshape(x[0:2],(2,1))
		qdot = agnp.reshape(x[2:],(2,1))
		th = x[1]
		thdot = x[3]

		# EOM from learning+control@caltech
		# D = agnp.array([[m_c+m_p,m_p*l*agnp.cos(th)],[m_p*l*agnp.cos(th),m_p*(l**2)]])
		a = m_c+m_p
		b = m_p*l*agnp.cos(th)
		c = m_p*l*agnp.cos(th)
		d = m_p*(l**2)
		Dinv = 1/(a*d-b*c) * agnp.array([[d, -b], [-c, a]])

		C = agnp.array([[0,-m_p*l*thdot*agnp.sin(th)],[0,0]])
		G = agnp.array([[0],[-m_p*g*l*agnp.sin(th)]])
		B = agnp.array([[1],[0]])
		qdotdot = agnp.dot(Dinv, agnp.dot(B,u) - agnp.dot(C,qdot) - G)

		res = agnp.vstack([qdot, qdotdot])
		return agnp.squeeze(res)

	def next_state(self, x, u):
		dt = self.ave_dt #self.times[self.time_step+1]-self.times[self.time_step]
		xdot = self.f(x, u)
		# euler integration
		sp1 = np.squeeze(np.reshape(x,(len(x),1)) + xdot * dt)
		# if sp1[1] > pi:
		# 	sp1[1] -= 2*pi
		# if sp1[1] < -pi:
		# 	sp1[1] += 2*pi

		return sp1

	def visualize(self,states,dt):

		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		import time

		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		vis["cart"].set_object(g.Box([0.2,0.5,0.2]))
		vis["pole"].set_object(g.Cylinder(self.length_pole, 0.01))

		while True:
			for state in states:
				vis["cart"].set_transform(tf.translation_matrix([0, state[0], 0]))

				vis["pole"].set_transform(
					tf.translation_matrix([0, state[0] + self.length_pole/2, 0]).dot(
					tf.rotation_matrix(np.pi/2 + state[1], [1,0,0], [0,-self.length_pole/2,0])))

				time.sleep(dt)

	def env_barrier(self,action):
		pass