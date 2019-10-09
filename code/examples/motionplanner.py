from param import Param
from run import run
from systems.motionplanner import MotionPlanner

class MotionPlannerParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'MotionPlanner'
		self.env_case = None

		# RL
		self.rl_train_model_fn = 'rl_model.pt'

		# IL
		self.il_train_model_fn = 'il_model.pt'
		self.il_imitate_model_fn = 'rl_model.pt'

		# Sim
		self.sim_rl_model_fn = 'rl_model.pt'
		self.sim_il_model_fn = 'il_model.pt'


if __name__ == '__main__':
	param = MotionPlannerParam()
	env = MotionPlanner(param)
	run(param, env)
