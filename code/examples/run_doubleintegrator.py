
from param import Param
from run import run
from systems.doubleintegrator import DoubleIntegrator
from other_policy import CBF

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple

class DoubleIntegratorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'DoubleIntegrator'
		self.env_case = None

		# flags
		self.pomdp_on = True
		self.single_agent_sim = False
		self.multi_agent_sim = True
		self.il_state_loss_on = False
		self.sim_render_on = False		

		# orca param
		self.n_agents = 7
		self.r_comm = 2.0
		self.r_agent = 0.2
		# self.a_min = np.array([-2.0,-2.0]) # m/s
		# self.a_max = np.array([2.0,2.0]) # m/s
		self.a_max = 2
		self.a_min = -1*self.a_max
		self.phi_max = self.a_max
		self.phi_min = -1*self.phi_max
		self.v_max = 0.5
		self.v_min = -1*self.v_max
		self.r_safe = 2*self.r_agent
		self.max_neighbors = 10
		
		# other
		self.sim_t0 = 0
		self.sim_tf = 30
		self.sim_dt = .1
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# cbf 
		self.cbf_kp = 0.1
		self.cbf_kv = 0.5
		self.cbf_noise = 0.075

		# IL
		self.il_train_model_fn = '../models/doubleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/doubleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.8
		self.il_batch_size = 5000
		self.il_n_epoch = 500
		self.il_lr = 5e-3
		self.il_n_data = 100000
		self.il_log_interval = 1
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Empty' # 'Empty','Barrier'
		self.controller_learning_module = 'DeepSet' # 

		# learning hyperparameters
		n,m,h = 4,2,128 # state dim, action dim, hidden layer
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(n,h),
			nn.Linear(h,h),
			nn.Linear(h,h)])
		self.il_rho_network_architecture = nn.ModuleList([
			nn.Linear(h,h),
			nn.Linear(h,h),
			nn.Linear(h,m)])

		self.il_psi_network_architecture = nn.ModuleList([
			nn.Linear(m+m,h),
			nn.Linear(h,h),
			nn.Linear(h,m)])

		self.il_network_activation = tanh 

		# Sim
		self.sim_rl_model_fn = '../models/doubleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/doubleintegrator/il_current.pt'
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)

		# Barrier function stuff
		self.b_gamma = 0.05 # 0.05 



if __name__ == '__main__':
	param = DoubleIntegratorParam()
	env = DoubleIntegrator(param)

	controllers = {
		# 'IL':	torch.load(param.sim_il_model_fn),
		'CBF': CBF(param,env)
		# 'RL': torch.load(param.sim_rl_model_fn)
	}

	if True:
		InitialState = namedtuple('InitialState', ['start', 'goal'])

		s0 = np.zeros((env.n))
		r = 4.
		d_rad = 2*np.pi/env.n_agents
		for i in range(env.n_agents):
			idx = env.agent_idx_to_state_idx(i) + \
					np.arange(0,2)
			s0[idx] = np.array([r*np.cos(d_rad*i),r*np.sin(d_rad*i)]) \
			+ 0.001*np.random.random(size=(1,2))
		s0 = InitialState._make((s0, -s0))

		# import yaml
		# with open("../baseline/centralized-planner/examples/swap2.yaml") as map_file:
		# 	map_data = yaml.load(map_file)

		# s = []
		# g = []
		# for agent in map_data["agents"]:
		# 	s.extend(agent["start"])
		# 	s.extend([0,0])
		# 	g.extend(agent["goal"])
		# 	g.extend([0,0])

		# InitialState = namedtuple('InitialState', ['start', 'goal'])
		# s0 = InitialState._make((np.array(s), np.array(g)))

	else:
		s0 = env.reset()

	run(param, env, controllers, s0)