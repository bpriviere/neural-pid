
# Creating OpenAI gym Envs 

from gym import Env
import autograd.numpy as np  # Thinly-wrapped numpy
import autograd.numpy as agnp  # Thinly-wrapped numpy
from scipy.spatial.transform import Rotation as R

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

		# control bounds
		self.a_min = param.a_min
		self.a_max = param.a_max

		# initial conditions
		if param.env_case is 'SmallAngle':
			s = np.zeros(18)
			s[6:15] = R.from_euler('xyz', [5,2.5,0], degrees=True).as_dcm().flatten()
			self.init_state_start = s
			self.init_state_disturbance = np.zeros(18)
			self.env_state_bounds = np.ones(self.n)

			self.s_min = np.array( \
						[-2, -2, -2, \
						  -4, -4, -4, \
						  -1.001, -1.001, -1.001, -1.001, -1.001, -1.001, -1.001, -1.001, -1.001,
						  -50, -50, -50])
			self.s_max = -self.s_min
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
		self.alpha_w = 0.10
		self.alpha_a = 0.05
		self.alpha_R = 0.50
		self.alpha_v = 0.0
		self.max_reward = 0.5

		self.states_name = [
			'Position X [m]',
			'Position Y [m]',
			'Position Z [m]',
			'Velocity X [m/s]',
			'Velocity Y [m/s]',
			'Velocity Z [m/s]',
			'R11',
			'R11',
			'R12',
			'R21',
			'R21',
			'R22',
			'R31',
			'R31',
			'R32',
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
		ew = np.linalg.norm(self.s[15:18] - state_ref[15:18])
		R = self.s[6:15].reshape((3,3))
		eR = np.arccos((np.trace(R)-1) / 2)

		cost = (ep \
			 + self.alpha_v * ev \
			 + self.alpha_w * ew \
			 + self.alpha_a * np.linalg.norm(a) \
			 + self.alpha_R * eR) * self.ave_dt
		if cost > self.max_reward:
			print("warning: max reward too small", cost)
		return self.max_reward - cost

	def reset(self, initial_state = None):
		if initial_state is None:
			# while True:
			# 	rotation = R.random()
			# 	rpy = rotation.as_euler('xyz', degrees=True)
			# 	print(rpy)
			# 	if abs(rpy[0]) < 5 and abs(rpy[1]) < 5:
			# 		break
			roll = 30 #np.random.uniform(-10, 10)
			pitch = 30# np.random.uniform(-10, 10)
			yaw = 30 #np.random.uniform(-10, 10)
			rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
			self.s = np.zeros(18)
			# self.s[0:3] = np.random.uniform(-1, 1, 3)
			self.s[0:3] = np.array([-0.1,0.1,0.2])
			self.s[6:15] = rotation.as_dcm().flatten()
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
		omega = s[15:]
		R = s[6:15].reshape((3,3))

		# get input 
		a = np.reshape(a,(self.m,))
		eta = np.dot(self.B0,a)
		f_u = np.array([0,0,eta[0]])
		tau_u = np.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		dsdt[0:3] = s[3:6] 
		# mv = mg + R f_u 
		dsdt[3:6] = self.g + np.dot(R,f_u) / self.mass

		# dot{R} = R S(w)
		# to integrate the dynamics, we essentially need to apply
		# Rodriguez formula, see
		# https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf, 6.1.2
		# and https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
		# a) rotate w from body to world frame
		omega_world = R @ omega
		# b) apply Rodriguez formula
		omega_norm = np.linalg.norm(omega_world)
		if omega_norm > 0:
			wx, wy, wz = omega_world.flatten()
			K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
			rot_angle = omega_norm * self.ave_dt
			dRdt = np.eye(3) + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
			Rnew = dRdt @ R
			# c) re-orthogonolize Rnew using SVD, see sim-to-real paper
			if self.time_step % 100 == 0:
				u, s, v = np.linalg.svd(Rnew)
				Rnew = u @ v
			# print("Rnew", Rnew)

			# d) transform Rnew to a "delta R" that works with the usual euler integration
			dsdt[6:15] = ((Rnew - R) / self.ave_dt).flatten()

		# mJ = Jw x w + tau_u 
		dsdt[15:] = self.inv_J * (np.cross(self.J * omega,omega) + tau_u)
		return dsdt.reshape((len(dsdt),1))


	def f_scp(self,s,a):
		# input:
		# 	s, nd array, (n,)
		# 	a, nd array, (m,1)
		# output
		# 	dsdt, nd array, (n,1)

		dsdt = agnp.zeros(self.n)
		omega = s[15:].reshape((3,1))
		R = s[6:15].reshape((3,3))

		# get input 
		a = agnp.reshape(a,(self.m,1))
		eta = agnp.dot(self.B0,a)
		f_u = agnp.array([[0],[0],[eta[0]]])
		tau_u = agnp.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		dpdt = s[3:6] 
		# mv = mg + R f_u 
		dvdt = agnp.squeeze( self.g + agnp.dot(R,f_u) / self.mass )

		# dot{R} = R S(w)
		# to integrate the dynamics, we essentially need to apply
		# Rodriguez formula, see
		# https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf, 6.1.2
		# and https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
		# a) rotate w from body to world frame
		omega_world = agnp.dot(R, omega)
		# b) apply Rodriguez formula
		omega_norm = agnp.linalg.norm(omega_world)
		dRdt = agnp.zeros(9)
		if omega_norm > 0:
			wx, wy, wz = omega_world.flatten()
			K = agnp.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
			rot_angle = omega_norm * self.ave_dt
			dRdt = agnp.eye(3) + agnp.sin(rot_angle) * K + (1. - agnp.cos(rot_angle)) * agnp.dot(K,K)
			Rnew = agnp.dot(dRdt, R)
			# c) re-orthogonolize Rnew using SVD, see sim-to-real paper
			if self.time_step % 2 == 0:
				u, s, v = agnp.linalg.svd(Rnew)
				Rnew = u @ v

			# d) transform Rnew to a "delta R" that works with the usual euler integration
			dRdt = ((Rnew - R) / self.ave_dt).flatten()

		# mJ = Jw x w + tau_u 
		dwdt = agnp.squeeze( agnp.dot(self.inv_J, 
			agnp.reshape(np.cross(agnp.squeeze(agnp.dot(self.J,omega)),agnp.squeeze(omega)),(3,1)) + tau_u) )
		return agnp.concatenate((dpdt, dvdt, dRdt, dwdt))


	def next_state(self,s,a):
		dt = self.times[self.time_step+1]-self.times[self.time_step]
		dsdt = self.f(s,a)
		sp1 = np.squeeze(np.reshape(s,(len(s),1)) + dsdt*dt)
		return sp1

	def deduce_state(self, s):
		rotation = s[6:15].reshape((3,3))
		rpy = R.from_dcm(rotation).as_euler('xyz', degrees=True)
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
				rotation = state[6:15].reshape((3,3))
				q = R.from_dcm(rotation).as_quat()
				vis["Quadrotor"].set_transform(
					tf.translation_matrix([state[0], state[1], state[2]]).dot(
					  tf.quaternion_matrix(q)))
				time.sleep(dt)

	def env_barrier(self,action):
		pass