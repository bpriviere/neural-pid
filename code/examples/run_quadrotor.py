from param import Param
from run import run
from systems.quadrotor import Quadrotor
import numpy as np
import matplotlib.pyplot as plt
import rowan
import torch

# load module that contains the CF firmware as baseline
import os, sys
sys.path.insert(1, os.path.join(os.getcwd(),'../baseline'))
import cfsim.cffirmware as firm

class QuadrotorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'Quadrotor'
		self.env_case = 'SmallAngle'

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

		self.rl_continuous_on = False
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
		self.sim_rl_model_fn = '../models/quadrotor/rl_current.pt'
		self.sim_il_model_fn = '../models/quadrotor/il_current.pt'
		self.sim_render_on = False

		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.01
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)

		s_desired = np.zeros(13)
		s_desired[6:10] = rowan.from_euler(np.radians(0), np.radians(0), np.radians(0), 'xyz')
		self.ref_trajectory = np.tile(np.array([s_desired.T]).T, (1, self.sim_nt))

		self.pomdp_on = False
		self.single_agent_sim = True
		self.multi_agent_sim = False


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
		'RL':	torch.load(param.rl_train_model_fn),
		'FW':	FirmwareController(param.a_min, param.a_max),
		# 'RRT':	FilePolicy(param.rrt_fn),
		# 'SCP':	FilePolicy(param.scp_fn),
	}

	run(param, env, controllers)

	# q = np.array(controllers['FW'].q)
	# qr = np.array(controllers['FW'].qr)
	# omega = np.array(controllers['FW'].omega)
	# omegar = np.array(controllers['FW'].omegar)


	# fig, ax = plt.subplots(2, 3)
	# for i in range(3):
	# 	ax[0][i].plot(omega[:,i],label='omega' + str(i))
	# 	ax[0][i].plot(omegar[:,i], label='omegar' + str(i))
	# 	ax[0][i].legend()
	# 	ax[1][i].plot(q[:,i],label='q' + str(i))
	# 	ax[1][i].plot(qr[:,i], label='qr' + str(i))
	# 	ax[1][i].legend()

	# plt.show()
