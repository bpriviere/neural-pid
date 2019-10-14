from param import Param
from run import run
from systems.quadrotor import Quadrotor
from numpy import array, zeros 

class QuadrotorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'Quadrotor'
		self.env_case = 'SmallAngle'

		self.mass = array([
			[1,1,1],
			[1,1,1],
			[1,1,1]
			])
		self.J = array([
			[1,1,1],
			[1,1,1],
			[1,1,1]
			])
		self.c_T = 1.
		self.c_Q = 1.
		self.l_a = 1.
		self.g = 9.81 # not signed

		self.ref_trajectory = zeros((18,self.sim_nt))

		# RL
		self.rl_train_model_fn = '../models/quadrotor/rl_current.pt'

		# IL
		self.il_train_model_fn = '../models/quadrotor/il_current.pt'
		self.il_imitate_model_fn = '../models/quadrotor/rl_current.pt'

		# Sim
		self.sim_rl_model_fn = '../models/quadrotor/rl_current.pt'
		self.sim_il_model_fn = '../models/quadrotor/il_current.pt'
		self.sim_render_on = True

		self.states_name = [
			'Position X [m]',
			'Position Y [m]',
			'Position Z [m]',
			'Velocity X [m/s]',
			'Velocity Y [m/s]',
			'Velocity Z [m/s]',
			'Angle X [rad]',
			'Angle Y [rad]',
			'Angle Z [rad]',
			'Anglular Velocity X [rad/s]',
			'Anglular Velocity Y [rad/s]',
			'Anglular Velocity Z [rad/s]']
		self.actions_name = [
			'Motor Speed 1',
			'Motor Speed 2'
			'Motor Speed 3'
			'Motor Speed 4']


if __name__ == '__main__':
	param = QuadrotorParam()
	env = Quadrotor(param)
	run(param, env)
