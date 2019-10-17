
from param import Param
from run import run
from systems.consensus import Consensus

import numpy as np 

class ConsensusParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'Consensus'
		self.env_case = None

		#
		self.pomdp_on = True
		self.r_comm = 2.2
		self.n_agents = 5
		self.n_malicious = 1

		# RL
		self.rl_train_model_fn = '../models/consensus/rl_current.pt'

		# IL
		self.il_train_model_fn = '../models/consensus/il_current.pt'
		self.il_imitate_model_fn = '../models/consensus/rl_current.pt'

		# Sim
		self.sim_rl_model_fn = '../models/consensus/rl_current.pt'
		self.sim_il_model_fn = '../models/consensus/il_current.pt'
		self.sim_render_on = True


if __name__ == '__main__':
	param = ConsensusParam()
	env = Consensus(param)
	run(param, env)
