
from param import Param
from run import run
from systems.singleintegrator import SingleIntegrator

# standard
from torch import nn, tanh
import torch
import numpy as np 

class SingleIntegratorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'SingleIntegrator'
		self.env_case = None

		# flags
		self.pomdp_on = True
		self.single_agent_sim = False
		self.multi_agent_sim = True
		self.il_state_loss_on = False
		self.sim_render_on = False		

		# orca param
		self.n_agents = 5
		self.r_comm = 15
		self.r_agent = 1.5
		self.sim_dt = 0.25
		self.a_min = np.array([-2.0,-2.0]) # m/s
		self.a_max = np.array([2.0,2.0]) # m/s

		# other
		self.sim_tf = 100

		# learning hyperparameters
		n,m,h = 4,2,32 # state dim, action dim, hidden layer
		self.rl_phi_network_architecture = nn.ModuleList([
			nn.Linear(n,h),
			nn.Linear(h,h)])
		self.rl_rho_network_architecture = nn.ModuleList([
			nn.Linear(h+n,h+n),
			nn.Linear(h+n,m)])
		self.rl_network_activation = tanh 

		# RL
		self.rl_train_model_fn = '../models/singleintegrator/rl_current.pt'

		# IL
		self.il_train_model_fn = '../models/singleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/singleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.8
		self.il_batch_size = 500
		self.il_n_data = 10000

		# Controller
		self.controller_class = 'Barrier' # 'Empty','Barrier','PID',
		self.controller_learning_module = 'DeepSet' # 

		# Sim
		self.sim_rl_model_fn = '../models/singleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/singleintegrator/il_current.pt'
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)


if __name__ == '__main__':
	param = SingleIntegratorParam()
	env = SingleIntegrator(param)

	controllers = {
		'IL':	torch.load(param.sim_il_model_fn),
		# 'RL': torch.load(param.sim_rl_model_fn)
	}
	run(param, env, controllers)
