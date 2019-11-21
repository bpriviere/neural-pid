from param import Param
from run import run
from systems.motionplanner import MotionPlanner

import numpy as np 
import torch 


class MotionPlannerParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'MotionPlanner'
		self.env_case = None

		# flags
		self.pomdp_on = True 

		# RL
		self.rl_train_model_fn = '../models/motionplanner/rl_current.pt'

		# IL
		self.il_train_model_fn = '../models/motionplanner/il_current.pt'
		self.il_imitate_model_fn = '../models/motionplanner/rl_current.pt'
		self.kp = 2
		self.kd = 5

		# Sim
		self.sim_rl_model_fn = '../models/motionplanner/rl_current.pt'
		self.sim_il_model_fn = '../models/motionplanner/il_current.pt'

		self.sim_render_on = True
		self.sim_t0 = 0
		self.sim_tf = 10
		self.sim_dt = 0.05
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)


if __name__ == '__main__':
	param = MotionPlannerParam()
	env = MotionPlanner(param)

	x0 = env.reset()

	controllers = {
		'RL':	torch.load(param.sim_rl_model_fn),
		'IL':	torch.load(param.sim_il_model_fn),
	}

	run(param, env, controllers, x0)
