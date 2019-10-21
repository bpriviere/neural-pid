
# Creating OpenAI gym Envs 

from gym import Env
import autograd.numpy as np  # Thinly-wrapped numpy
import autograd.numpy as agnp  # Thinly-wrapped numpy
import rowan

# Quaternion routines adapted from rowan to use autograd
def qmultiply(q1, q2):
	return agnp.concatenate((
		agnp.array([q1[0] * q2[0]]), # w1w2
		q1[0] * q2[1:4] + q2[0] * q1[1:4] + agnp.cross(q1[1:4], q2[1:4])))

def qconjugate(q):
	return agnp.concatenate((q[0:1],-q[1:4]))

def qrotate(q, v):
	quat_v = agnp.concatenate((agnp.array([0]), v))
	return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]

def qexp(q):
	norm = agnp.linalg.norm(q[1:4])
	e = agnp.exp(q[0])
	result_w = e * agnp.cos(norm)
	if agnp.isclose(norm, 0):
		result_v = agnp.zeros(3)
	else:
		result_v = e * q[1:4] / norm * agnp.sin(norm)
	return agnp.concatenate((agnp.array([result_w]), result_v))

def qintegrate(q, v, dt):
	quat_v = agnp.concatenate((agnp.array([0]), v*dt/2))
	return qmultiply(qexp(quat_v), q)

def qnormalize(q):
	return q / agnp.linalg.norm(q)

class Quadrotor(Env):

	def __init__(self, param):

		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None
		self.ave_dt = self.times[1]-self.times[0]
		self.param = param

		# dim 
		self.n = 13
		self.m = 4

		# control bounds
		self.a_min = param.a_min
		self.a_max = param.a_max

		# initial conditions
		if param.env_case is 'SmallAngle':
			self.s_min = np.array( \
						[-10, -10, -10, \
						  -4, -4, -4, \
						  -1.001, -1.001, -1.001, -1.001,
						  -50, -50, -50])
			self.s_max = -self.s_min
			self.rpy_limit = np.array([60, 60, 60])
		else:
			raise Exception('param.env_case invalid ' + param.env_case)

		# parameters
		self.mass = param.mass
		self.J = self.param.J
		self.g = np.array([0,0,-param.g])
		self.inv_mass = 1 / self.mass
		if self.J.shape == (3,3):
			self.inv_J = np.linalg.pinv(self.J) # full matrix -> pseudo inverse
		else:
			self.inv_J = 1 / self.J # diagonal matrix -> division
		self.B0 = param.B0

		# reward function stuff
		# see row 8, Table 3, sim-to-real paper
		self.alpha_p = 0.01 #1.0
		self.alpha_w = 0.0 #0.10
		self.alpha_a = 0 #0.05
		self.alpha_R = 0.1 #0.50
		self.alpha_v = 0.0 #0.0

		max_p_error = np.linalg.norm(self.s_max[0:3] - self.s_min[0:3])
		max_w_error = np.linalg.norm(self.s_max[10:13] - self.s_min[10:13])
		max_v_error = np.linalg.norm(self.s_max[3:6] - self.s_min[3:6])
		max_R_error = np.sqrt(2)
		self.max_reward = self.alpha_R * max_R_error \
						+ self.alpha_p * max_p_error \
						+ self.alpha_w * max_w_error \
						+ self.alpha_v * max_v_error

		self.states_name = [
			'Position X [m]',
			'Position Y [m]',
			'Position Z [m]',
			'Velocity X [m/s]',
			'Velocity Y [m/s]',
			'Velocity Z [m/s]',
			'qw',
			'qx',
			'qy',
			'qz',
			'Angular Velocity X [rad/s]',
			'Angular Velocity Y [rad/s]',
			'Angular Velocity Z [rad/s]']

		self.deduced_state_names = [
			'Roll [deg]',
			'Pitch [deg]',
			'Yaw [deg]',
		]


		self.actions_name = [
			'Motor Force 1 [N]',
			'Motor Force 2 [N]',
			'Motor Force 3 [N]',
			'Motor Force 4 [N]']


	def step(self,a):
		self.s = self.next_state(self.s,a)
		d = self.done() 
		r = self.reward(a)
		self.time_step += 1
		return self.s, r, d, {}


	def done(self):
		if (self.s < self.s_min).any() or (self.s > self.s_max).any():
			return True
		return False


	def reward(self,a):
		# see sim-to-real paper, eq (14)
		state_ref = self.param.ref_trajectory[:,self.time_step]
		ep = np.linalg.norm(self.s[0:3] - state_ref[0:3])
		ev = np.linalg.norm(self.s[3:6] - state_ref[3:6])
		ew = np.linalg.norm(self.s[10:13] - state_ref[10:13])
		# R = self.s[6:15].reshape((3,3))
		# eR = np.arccos((np.trace(R)-1) / 2)
		eR = rowan.geometry.sym_distance(self.s[6:10], np.array([1,0,0,0]))

		cost = (self.alpha_p * ep \
			 + self.alpha_v * ev \
			 + self.alpha_w * ew \
			 + self.alpha_a * np.linalg.norm(a) \
			 + self.alpha_R * eR) * self.ave_dt
		if cost > self.max_reward:
			print("warning: max reward too small", cost)
		return self.max_reward - cost

	def reset(self, initial_state = None):
		if initial_state is None:
			self.s = np.empty(self.n)
			# position and velocity
			limits = np.array([0.5,0.5,0.5,1,1,1, 0, 0, 0, 0, 12, 12, 12])
			self.s[0:6] = np.random.uniform(-limits[0:6], limits[0:6], 6)
			# rotation
			rpy = np.radians(np.random.uniform(-self.rpy_limit, self.rpy_limit, 3))
			q = rowan.from_euler(rpy[0], rpy[1], rpy[2], 'xyz')
			self.s[6:10] = q
			# angular velocity
			self.s[10:13] = np.random.uniform(-limits[10:13], limits[10:13], 3)
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
		q = s[6:10]
		omega = s[10:]

		# get input 
		a = np.reshape(a,(self.m,))
		eta = np.dot(self.B0,a)
		f_u = np.array([0,0,eta[0]])
		tau_u = np.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		dsdt[0:3] = s[3:6] 
		# mv = mg + R f_u 
		dsdt[3:6] = self.g + rowan.rotate(q,f_u) / self.mass

		# dot{R} = R S(w)
		# to integrate the dynamics, see
		# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
		# https://arxiv.org/pdf/1604.08139.pdf
		qnew = rowan.calculus.integrate(q, omega, self.ave_dt)
		qnew = rowan.normalize(qnew)
		if qnew[0] < 0:
			qnew = -qnew
		# transform qnew to a "delta q" that works with the usual euler integration
		dsdt[6:10] = (qnew - q) / self.ave_dt

		# mJ = Jw x w + tau_u 
		dsdt[10:] = self.inv_J * (np.cross(self.J * omega,omega) + tau_u)
		return dsdt.reshape((len(dsdt),1))


	def f_scp(self,s,a):
		# input:
		# 	s, nd array, (n,)
		# 	a, nd array, (m,1)
		# output
		# 	dsdt, nd array, (n,1)

		dsdt = agnp.zeros(self.n)
		q = s[6:10]
		omega = s[10:]

		# get input 
		a = agnp.reshape(a,(self.m,))
		eta = agnp.dot(self.B0,a)
		f_u = agnp.array([0,0,eta[0]])
		tau_u = agnp.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		dpdt = s[3:6] 
		# mv = mg + R f_u 
		dvdt = self.g + qrotate(q,f_u) / self.mass

		# dot{R} = R S(w)
		# to integrate the dynamics, see
		# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
		# https://arxiv.org/pdf/1604.08139.pdf
		qnew = qintegrate(q, omega, self.ave_dt)
		qnew = qnormalize(qnew)
		# transform qnew to a "delta q" that works with the usual euler integration
		dqdt = (qnew - q) / self.ave_dt

		# mJ = Jw x w + tau_u 
		dwdt = self.inv_J * (agnp.cross(self.J * omega,omega) + tau_u)
		return agnp.concatenate((dpdt, dvdt, dqdt, dwdt))


	def next_state(self,s,a):
		dt = self.times[self.time_step+1]-self.times[self.time_step]
		dsdt = self.f(s,a)
		sp1 = np.squeeze(np.reshape(s,(len(s),1)) + dsdt*dt)
		return sp1

	def deduce_state(self, s):
		rpy = np.degrees(rowan.to_euler(rowan.normalize(s[6:10]), 'xyz'))
		return rpy

	def sample_state_around(self, s):
		dp = np.random.normal(0, 0.05, 3)
		dv = np.random.normal(0, 0.1, 3)
		rpy = np.random.normal(0, 5, 3)
		dR = R.from_euler('xyz', rpy, degrees=True).as_dcm()
		dw = np.random.normal(0, 0.5, 3)

		result = np.concatenate((s[0:3] + dp, s[3:6] + dv, (s[6:15].reshape((3,3)) @ dR).flatten(), s[15:18] + dw))
		return np.clip(result, self.s_min, self.s_max)

	def visualize(self,states,dt):

		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		import time 

		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		vis["/Cameras/default"].set_transform(
			tf.translation_matrix([0,0,0]).dot(
			tf.euler_matrix(0,np.radians(-30),-np.pi/2)))

		vis["/Cameras/default/rotated/<object>"].set_transform(
			tf.translation_matrix([1, 0, 0]))

		vis["Quadrotor"].set_object(g.StlMeshGeometry.from_file('systems/crazyflie2.stl'))

		while True:
			for state in states:
				vis["Quadrotor"].set_transform(
					tf.translation_matrix([state[0], state[1], state[2]]).dot(
					  tf.quaternion_matrix(state[6:10])))
				time.sleep(dt)

	def env_barrier(self,action):
		pass