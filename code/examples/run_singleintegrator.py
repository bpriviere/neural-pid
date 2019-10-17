
from param import Param
from run import run
from systems.singleintegrator import SingleIntegrator
from torch import nn, tanh
import torch

class SingleIntegratorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'SingleIntegrator'
		self.env_case = None

		# flags
		self.pomdp_on = True
		self.single_agent_sim = False
		self.multi_agent_sim = True

		# param
		self.n_agents = 2
		self.r_comm = 15

		# learning hyperparameters
		n,m,h = 4,2,32 # state dim, action dim, hidden layer
		self.network_architecture_phi = nn.ModuleList([
			nn.Linear(n,h), 
			nn.Linear(h,h)])
		self.network_architecture_rho = nn.ModuleList([
			nn.Linear(h+2*n,h+2*n), 
			nn.Linear(h+2*n,m)])
		self.network_activation = tanh 

		# RL
		self.rl_train_model_fn = '../models/singleintegrator/rl_current.pt'

		# IL
		self.il_train_model_fn = '../models/singleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/singleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.8
		self.il_batch_size = 200 
		self.il_n_data = 5000

		# Controller
		self.controller_class = 'DeepSet'

		# Sim
		self.sim_rl_model_fn = '../models/singleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/singleintegrator/il_current.pt'
		self.sim_render_on = False


if __name__ == '__main__':
	param = SingleIntegratorParam()
	env = SingleIntegrator(param)

	controllers = {
		'IL':	torch.load(param.sim_il_model_fn),
	}
	run(param, env, controllers)
