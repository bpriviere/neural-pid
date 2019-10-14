
# Creating OpenAI gym Envs 

from gym import Env
import autograd.numpy as np  # Thinly-wrapped numpy

class Quadrotor(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None
		self.ave_dt = self.times[1]-self.times[0]
		self.param = param

		# dim 
		self.n = 18
		self.m = 4

		# initial conditions
		if param.env_case is 'SmallAngle':
			self.init_state_start = np.array([
				0,0,0,0,0,0,np.pi/6,0,0,0,0,0,0,0,0,0,0,0
				])
			self.init_state_disturbance = 0.01*np.ones(self.n)
			self.env_state_bounds = np.ones(self.n)
		else:
			raise Exception('param.env_case invalid ' + param.env_case)

		# parameters
		self.mass = self.param.mass 
		self.J = self.param.J
		self.g = np.array([0,0,-self.param.g],ndmin=2).T
		self.inv_mass = np.linalg.pinv(self.mass)
		self.inv_J = np.linalg.pinv(self.J)
		self.B0 = np.array([
			[self.param.c_T, self.param.c_T, self.param.c_T, self.param.c_T],
			[0,self.param.c_T*self.param.l_a,0,-self.param.c_T*self.param.l_a],
			[-self.param.c_T*self.param.l_a,0,self.param.c_T*self.param.l_a,0],
			[-self.param.c_Q,self.param.c_Q,-self.param.c_Q,self.param.c_Q]
			])

		# reward function stuff
		self.W = np.eye(self.n)
		self.max_reward = 1 
		self.max_error = 2*self.env_state_bounds
		self.max_penalty = np.dot(self.max_error.T,np.dot(self.W,self.max_error))		



	def step(self,a):
		self.s = self.next_state(self.s,a)
		d = self.done() 
		r = self.reward()
		self.time_step += 1
		return self.s, r, d, {}


	def done(self):
		if any( np.abs(self.s) > self.env_state_bounds):
			return True
		return False


	def reward(self):
		state_ref = self.param.ref_trajectory[:,self.time_step]
		error = self.s - state_ref
		return 1 - np.power(np.dot(error.T,np.dot(self.W,error))/self.max_penalty,1/6)

		
	def reset(self, initial_state = None):
		if initial_state is None:
			self.s = self.init_state_start+\
			np.multiply(self.init_state_disturbance, np.random.uniform(size=(self.n,)))
		else:
			self.s = initial_state
		self.time_step = 0
		return np.array(self.s)


	# dsdt = f(s,a)
	def f(self,s,a):
		# input:
		# 	s, nd array, (n,)
		# 	a, nd array, (m,1)
		# output
		# 	dsdt, nd array, (n,1)

		dsdt = np.zeros(self.n)
		omega = s[15:].reshape((3,1))
		R = self.R()
		S = self.S(omega)

		# get input 
		a = np.reshape(a,(self.m,1))
		eta = np.dot(self.B0,a)
		f_u = np.array([[0],[0],[eta[0]]])
		tau_u = np.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		dsdt[0:3] = s[3:6] 
		# mv = mg + R f_u 
		dsdt[3:6] = np.squeeze( np.dot(self.inv_mass, np.dot(self.mass,self.g) + np.dot(R,f_u)) )
		# dot{R} = R S(w) 
		dsdt[6:15] = np.dot(R,S).flatten()
		# mJ = Jw x w + tau_u 
		dsdt[15:] = np.squeeze( np.dot(self.inv_J, 
			np.reshape(np.cross(np.squeeze(np.dot(self.J,omega)),np.squeeze(omega)),(3,1)) + tau_u) )
		return dsdt.reshape((len(dsdt),1))

	def R(self):
		# rotation matrix 
		return np.array([
				self.s[6:9],
				self.s[9:12],
				self.s[12:15]
			])

	def S(self,w):
		# skew symmetric operation
		return np.array([
			[0,-w[2],w[1]],
			[w[2],0,-w[0]],
			[-w[1],w[0],0]
			])


	def f_scp(self,s,a):
		return np.squeeze(self.f(s,a))


	def next_state(self,s,a):
		dt = self.times[self.time_step+1]-self.times[self.time_step]
		dsdt = self.f(s,a)
		sp1 = np.squeeze(np.reshape(s,(len(s),1)) + dsdt*dt)
		return sp1


	def env_barrier(self,action):
		pass