from param import Param
from run import run
from systems.quadrotor import Quadrotor
import numpy as np
from scipy.spatial.transform import Rotation as R

class QuadrotorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'Quadrotor'
		self.env_case = 'SmallAngle'

		# Crazyflie 2.0 quadrotor
		self.mass = 0.034 # kg
		self.J = np.array([
			[16.56,0.83,0.71],
			[0.83,16.66,1.8],
			[0.72,1.8,29.26]
			]) * 1e-6  # kg m^2

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

		s_desired = np.zeros(18)
		s_desired[6:15] = R.from_euler('zyx', [0,0,0]).as_dcm().flatten()
		self.ref_trajectory = np.tile(np.array([s_desired.T]).T, (1, self.sim_nt))


if __name__ == '__main__':
	param = QuadrotorParam()
	env = Quadrotor(param)
	run(param, env)
