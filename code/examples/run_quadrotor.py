
# my package
from param import Param
from run import run
from systems.quadrotor import Quadrotor

# standard package
import numpy as np
import matplotlib.pyplot as plt
import rowan
import torch
from torch import nn,tanh

# load module that contains the CF firmware as baseline
import os, sys
sys.path.insert(1, os.path.join(os.getcwd(),'../baseline'))
import cfsim.cffirmware as firm

class QuadrotorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'Quadrotor'
		self.env_case = 'SmallAngle'

		# flags
		self.rl_continuous_on = True
		self.sim_render_on = False
		self.pomdp_on = False
		self.single_agent_sim = True
		self.multi_agent_sim = False

		# Crazyflie 2.0 quadrotor
		self.mass = 0.034 # kg
		# self.J = np.array([
		# 	[16.56,0.83,0.71],
		# 	[0.83,16.66,1.8],
		# 	[0.72,1.8,29.26]
		# 	]) * 1e-6  # kg m^2
		self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		t2t = 0.006 # thrust-to-torque ratio
		self.B0 = np.array([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])
		self.g = 9.81 # not signed

		# control limits [N]
		self.a_min = np.array([0, 0, 0, 0])
		self.a_max = np.array([12, 12, 12, 12]) / 1000 * 9.81 # g->N

		# perfect hover would use: np.array([0.0085, 0.0085, 0.0085, 0.0085]) * 9.81
		# self.a_min = np.array([0.008, 0.008, 0.008, 0.008]) * 9.81
		# self.a_max = np.array([0.012, 0.012, 0.012, 0.012]) * 9.81 # g->N

		# RL
		self.rl_train_model_fn = '../models/quadrotor/rl_current.pt'
		self.rl_lr_schedule_on = False
		self.rl_lr_schedule_gamma = 0.2
		self.rl_warm_start_on = False
		self.rl_warm_start_fn = '../models/quadrotor/rl_continuous_v3.pt'
		self.rl_module = 'DDPG'
		self.rl_lr_schedule = np.arange(0,10)
		self.rl_batch_size = 2000
		# common param
		self.rl_gamma = 0.999
		self.rl_K = 10
		self.rl_max_episodes = 50000
		self.rl_batch_size = 2000
		if self.rl_continuous_on:
			# ddpg param
			self.rl_lr_mu = 1e-4
			self.rl_lr_q = 1e-3
			self.rl_buffer_limit = 5e6
			self.rl_action_std = 0.05
			self.rl_max_action_perturb = 0.05
			self.rl_tau = 0.995
			# network architecture
			n,m,h_mu,h_q = 13,4,64,64 # state dim, action dim, hidden layers
			self.rl_mu_network_architecture = nn.ModuleList([
				nn.Linear(n,h_mu), 
				nn.Linear(h_mu,h_mu),
				nn.Linear(h_mu,m)])
			self.rl_q_network_architecture = nn.ModuleList([
				nn.Linear(n+m,h_q),
				nn.Linear(h_q,h_q),
				nn.Linear(h_q,1)])
			self.rl_network_activation = tanh 

		else:
			# ppo param s
			self.rl_lmbda = 0.95
			self.rl_eps_clip = 0.2
			self.rl_discrete_action_space = [
				np.array([0, 0, 0, 0]) * 12 / 1000 * 9.81,
				np.array([0, 0, 0, 1]) * 12 / 1000 * 9.81,
				np.array([0, 0, 1, 0]) * 12 / 1000 * 9.81,
				np.array([0, 0, 1, 1]) * 12 / 1000 * 9.81,
				np.array([0, 1, 0, 0]) * 12 / 1000 * 9.81,
				np.array([0, 1, 0, 1]) * 12 / 1000 * 9.81,
				np.array([0, 1, 1, 0]) * 12 / 1000 * 9.81,
				np.array([0, 1, 1, 1]) * 12 / 1000 * 9.81,
				np.array([1, 0, 0, 0]) * 12 / 1000 * 9.81,
				np.array([1, 0, 0, 1]) * 12 / 1000 * 9.81,
				np.array([1, 0, 1, 0]) * 12 / 1000 * 9.81,
				np.array([1, 0, 1, 1]) * 12 / 1000 * 9.81,
				np.array([1, 1, 0, 0]) * 12 / 1000 * 9.81,
				np.array([1, 1, 0, 1]) * 12 / 1000 * 9.81,
				np.array([1, 1, 1, 0]) * 12 / 1000 * 9.81,
				np.array([1, 1, 1, 1]) * 12 / 1000 * 9.81,
			]
			self.rl_lr = 1e-3 #5e-3

		# IL
		self.il_train_model_fn = '../models/quadrotor/il_current.pt'
		self.il_imitate_model_fn = '../models/quadrotor/rl_current.pt'

		# Sim
		self.sim_rl_model_fn = '../models/quadrotor/rl_discrete.pt' # rl_current
		self.sim_il_model_fn = '../models/quadrotor/il_current.pt'

		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.01
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)

		s_desired = np.zeros(13)
		s_desired[6:10] = rowan.from_euler(np.radians(0), np.radians(0), np.radians(0), 'xyz')
		self.ref_trajectory = np.tile(np.array([s_desired.T]).T, (1, self.sim_nt))



class FirmwareController:
	"""
	Controller that uses the actual firmware C-code
	"""
	def __init__(self, a_min, a_max):
		firm.controllerSJCInit()
		self.control = firm.control_t()
		self.setpoint = firm.setpoint_t()
		self.sensors = firm.sensorData_t()
		self.state = firm.state_t()

		# update setpoint
		self.setpoint.position.x = 0
		self.setpoint.position.y = 0
		self.setpoint.position.z = 0
		self.setpoint.velocity.x = 0
		self.setpoint.velocity.y = 0
		self.setpoint.velocity.z = 0
		self.setpoint.attitude.yaw = 0
		self.setpoint.attitudeRate.roll = 0
		self.setpoint.attitudeRate.pitch = 0
		self.setpoint.attitudeRate.yaw = 0
		self.setpoint.mode.x = firm.modeAbs
		self.setpoint.mode.y = firm.modeAbs
		self.setpoint.mode.z = firm.modeAbs
		self.setpoint.mode.roll = firm.modeDisable
		self.setpoint.mode.pitch = firm.modeDisable
		self.setpoint.mode.yaw = firm.modeDisable
		self.setpoint.mode.quat = firm.modeDisable
		self.setpoint.acceleration.x = 0
		self.setpoint.acceleration.y = 0
		self.setpoint.acceleration.z = 0

		self.tick = 0
		self.a_min = a_min
		self.a_max = a_max

		self.q = []
		self.qr = []
		self.omega = []
		self.omegar = []

	def policy(self, state):
		# set state
		self.state.position.x = state[0]
		self.state.position.y = state[1]
		self.state.position.z = state[2]

		self.state.velocity.x = state[3]
		self.state.velocity.y = state[4]
		self.state.velocity.z = state[5]

		rpy = np.degrees(rowan.to_euler(state[6:10], 'xyz'))
		self.state.attitude.roll = rpy[0]
		self.state.attitude.pitch = -rpy[1] # inverted coordinate system!
		self.state.attitude.yaw = rpy[2]

		self.sensors.gyro.x = np.degrees(state[10])
		self.sensors.gyro.y = np.degrees(state[11])
		self.sensors.gyro.z = np.degrees(state[12])

		firm.controllerSJC(self.control, self.setpoint, self.sensors, self.state, self.tick)
		self.tick += 10

		# power distribution
		thrust = self.control.thrustSI
		torqueArr = firm.floatArray_frompointer(self.control.torque)
		torque = np.array([torqueArr[0], torqueArr[1], torqueArr[2]])

		thrust_to_torque = 0.006
		arm_length = 0.046
		thrustpart = 0.25 * thrust
		yawpart = -0.25 * torque[2] / thrust_to_torque

		arm = 0.707106781 * arm_length
		rollpart = 0.25 / arm * torque[0]
		pitchpart = 0.25 / arm * torque[1]

		motorForce = np.array([
			thrustpart - rollpart - pitchpart + yawpart,
			thrustpart - rollpart + pitchpart - yawpart,
			thrustpart + rollpart + pitchpart + yawpart,
			thrustpart + rollpart - pitchpart - yawpart
		])
		motorForce = np.clip(motorForce, self.a_min, self.a_max)

		v = firm.controllerSJCGetq()
		self.q.append(np.array([v.x, v.y, v.z]))
		v = firm.controllerSJCGetqr()
		self.qr.append(np.array([v.x, v.y, v.z]))
		v = firm.controllerSJCGetomega()
		self.omega.append(np.array([v.x, v.y, v.z]))
		v = firm.controllerSJCGetomegar()
		self.omegar.append(np.array([v.x, v.y, v.z]))
		
		# return np.array([0.0, 0.01, 0.01, 0.0]) * 9.81

		return motorForce
		# return np.array([0.0085, 0.0085, 0.0085, 0.0085]) * 9.80

#   // For euler angles Z(q)^-1 is:
#   //           [1  0       -sin(p)     ]
#   // Z(q)^-1 = [0  cos(r)  cos(p)sin(r)]
#   //           [0 -sin(r)  cos(p)cos(r)]
def Zinv(q):
	return np.array([
		[1, 0, -np.sin(q[1])],
		[0, np.cos(q[0]), np.cos(q[1]) * np.sin(q[0])],
		[0, -np.sin(q[0]), np.cos(q[1]) * np.cos(q[0])]])


class SJCController:
	"""
	Controller proposed in 
	
	Daniel Morgan, Giri P Subramanian, Soon-Jo Chung, Fred Y Hadaegh
	Swarm assignment and trajectory optimization using variable-swarm, distributed auction assignment and sequential convex programming 
	IJRR 2016
	"""
	def __init__(self, J, mass, a_min, a_max):
		self.mass = mass
		self.J = J
		self.a_min = a_min
		self.a_max = a_max

		# desired state
		self.p_d = np.array([0,0,0])
		self.v_d = np.array([0,0,0])
		self.a_d = np.array([0,0,0 + 9.81])
		self.R_d = rowan.to_matrix([1, 0, 0, 0])
		self.omega_d = np.array([0,0,0])

		# Gains
		self.Kpos_P = np.array([10,10,5])
		self.Kpos_D = np.array([5,5,2.5])

		self.lambda_att = np.array([20,20,8])
		self.K_att = np.array([0.003, 0.003, 0.003])

		# state
		self.omega_r_last = None
		self.q_d_last = None

		self.q = []
		self.qr = []
		self.omega = []
		self.omegar = []

	def policy(self, state):
		# current state
		p = state[0:3]
		v = state[3:6]
		q = state[6:10]
		omega = state[10:13]
		R = rowan.to_matrix(q)

		# position controller
		pos_e = self.p_d - p
		vel_e = self.v_d - v
		F_d = self.mass * (self.a_d + self.Kpos_D * vel_e + self.Kpos_P * pos_e)
		print(F_d)

		thrust = np.linalg.norm(F_d)
		yaw = 0
		q_d = np.array([
			np.arcsin((F_d[0] * np.sin(yaw) - F_d[1] * np.cos(yaw)) / thrust),
			np.arctan((F_d[0] * np.cos(yaw) + F_d[1] * np.sin(yaw)) / F_d[2]),
			yaw])

		if self.q_d_last is not None:
			q_d_dot = (q_d - self.q_d_last) / 0.01
		else:
			q_d_dot = np.zeros(3)
		self.q_d_last = q_d

		# attitude controller
		q = rowan.to_euler(q, 'xyz')

		omega_r = Zinv(q) @ (q_d_dot +self.lambda_att * (q_d -q))
		if self.omega_r_last is not None:
			omega_r_dot = (omega_r - self.omega_r_last) / 0.01
		else:
			omega_r_dot = np.zeros(3)
		self.omega_r_last = omega_r

		torque = self.J * omega_r_dot \
				- np.cross(self.J * omega, omega_r) \
				- self.K_att * (omega - omega_r)

		# power distribution
		thrust_to_torque = 0.006
		arm_length = 0.046
		thrustpart = 0.25 * thrust
		yawpart = -0.25 * torque[2] / thrust_to_torque

		arm = 0.707106781 * arm_length
		rollpart = 0.25 / arm * torque[0]
		pitchpart = 0.25 / arm * torque[1]

		motorForce = np.array([
			thrustpart - rollpart - pitchpart + yawpart,
			thrustpart - rollpart + pitchpart - yawpart,
			thrustpart + rollpart + pitchpart + yawpart,
			thrustpart + rollpart - pitchpart - yawpart
		])
		motorForce = np.clip(motorForce, self.a_min, self.a_max)

		# logging
		self.qr.append(np.degrees(q_d))
		self.q.append(np.degrees(q))
		self.omegar.append(omega_r)
		self.omega.append(omega)

		return motorForce

def veemap(A):
	return np.array([A[2,1], -A[0,2], A[1,0]])

class XichenController:
	"""
	Controller proposed in "Nonlinear Control of Autonomous Flying Cars with Wings and
	Distributed Electric Propulsion", CDC 2018
	"""
	def __init__(self, J, mass, a_min, a_max):
		self.mass = mass
		self.J = J
		self.a_min = a_min
		self.a_max = a_max

		# desired state
		self.p_d = np.array([0,0,0])
		self.v_d = np.array([0,0,0])
		self.a_d = np.array([0,0,0 + 9.81])
		self.R_d = rowan.to_matrix([1, 0, 0, 0])
		self.omega_d = np.array([0,0,0])

		# Gains
		self.Kpos_P = np.array([10,10,5])
		self.Kpos_D = np.array([5,5,2.5])

		self.lambda_a = np.array([20,20,8])
		self.K_att = np.array([0.003, 0.003, 0.003])
		self.kq = 0

		# state
		self.omega_r_last = None

		self.q = []
		self.qr = []
		self.omega = []
		self.omegar = []

	def policy(self, state):
		# current state
		p = state[0:3]
		v = state[3:6]
		q = state[6:10]
		omega = state[10:13]
		R = rowan.to_matrix(q)

		# position controller
		# p_tilde = p - self.p_d
		# v_tilde = v - self.v_d
		# v_r = self.v_d - self.lambda_p * p_tilde
		# v_r_dot = self.a_d - self.lambda_p * v_tilde
		# s_v = v - v_r
		# f_r = self.mass * v_r_dot \
		# 	- self.Kv * s_v \
		# 	- self.Kp * p_tilde

		# thrust = np.linalg.norm(f_r)

		# qr = mkvec(
  #       asinf((F_d.x * sinf(yaw) - F_d.y * cosf(yaw)) / control->thrustSI),
  #       atanf((F_d.x * cosf(yaw) + F_d.y * sinf(yaw)) / F_d.z),
  #       desiredYaw);

		# position controller
		pos_e = self.p_d - p
		vel_e = self.v_d - v
		F_d = self.mass * (self.a_d + self.Kpos_D * vel_e + self.Kpos_P * pos_e)
		print(F_d)

		thrust = np.linalg.norm(F_d)
		yaw = 0
		rpy_d = np.array([
			np.arcsin((F_d[0] * np.sin(yaw) - F_d[1] * np.cos(yaw)) / thrust),
			np.arctan((F_d[0] * np.cos(yaw) + F_d[1] * np.sin(yaw)) / F_d[2]),
			yaw])
		R_d = rowan.to_matrix(rowan.from_euler(*rpy_d))
		print(rpy_d)
		print(R_d)

		# attitude controller

		# rotation error
		Rtilde = self.R_d.T @ R
		qtilde_0 = 1/2 * np.sqrt(1 + np.trace(Rtilde))
		qtilde_v = 1 / (4 * qtilde_0) * veemap(Rtilde - Rtilde.T)

		omega_r = Rtilde.T @ self.omega_d - 2 * self.lambda_a * qtilde_v
		print(omega_r)

		if self.omega_r_last is not None:
			omega_r_dot = (omega_r - self.omega_r_last) / 0.01
		else:
			omega_r_dot = np.zeros(3)
		self.omega_r_last = omega_r

		s_omega = omega - omega_r
		torque = self.J * omega_r_dot \
				- np.cross(self.J * omega, omega_r) \
				- self.K_att * s_omega \
				- self.kq * qtilde_v

		# power distribution
		thrust_to_torque = 0.006
		arm_length = 0.046
		thrustpart = 0.25 * thrust
		yawpart = -0.25 * torque[2] / thrust_to_torque

		arm = 0.707106781 * arm_length
		rollpart = 0.25 / arm * torque[0]
		pitchpart = 0.25 / arm * torque[1]

		motorForce = np.array([
			thrustpart - rollpart - pitchpart + yawpart,
			thrustpart - rollpart + pitchpart - yawpart,
			thrustpart + rollpart + pitchpart + yawpart,
			thrustpart + rollpart - pitchpart - yawpart
		])
		motorForce = np.clip(motorForce, self.a_min, self.a_max)

		# logging
		self.qr.append(np.degrees(rpy_d))
		self.q.append(np.degrees(rowan.to_euler(q, 'xyz')))
		self.omegar.append(omega_r)
		self.omega.append(omega)

		return motorForce

class FilePolicy:
	def __init__(self, filename):
		data = np.loadtxt(filename, delimiter=',', ndmin=2)
		self.states = data[:,0:13]
		self.actions = data[:,13:17]
		self.steps = data.shape[0]
		print(self.actions.shape)


if __name__ == '__main__':
	param = QuadrotorParam()
	env = Quadrotor(param)

	controllers = {
		# 'RL':	torch.load(param.sim_rl_model_fn),
		'FW':	FirmwareController(param.a_min, param.a_max),
		# 'FW SJC':	SJCController(param.mass, param.J, param.a_min, param.a_max),
		# 'RRT':	FilePolicy(param.rrt_fn),
		# 'SCP':	FilePolicy("/home/whoenig/projects/caltech/neural-pid/models/quadrotor/dataset_rl/scp_1.csv"),
	}

	x0 = None #controllers['SCP'].states[0]

	run(param, env, controllers, x0)

	q = np.array(controllers['FW SJC'].q)
	qr = np.array(controllers['FW SJC'].qr)
	omega = np.array(controllers['FW SJC'].omega)
	omegar = np.array(controllers['FW SJC'].omegar)


	fig, ax = plt.subplots(2, 3)
	for i in range(3):
		ax[0][i].plot(omega[:,i],label='omega' + str(i))
		ax[0][i].plot(omegar[:,i], label='omegar' + str(i))
		ax[0][i].legend()
		ax[1][i].plot(q[:,i],label='q' + str(i))
		ax[1][i].plot(qr[:,i], label='qr' + str(i))
		ax[1][i].legend()

	plt.show()
