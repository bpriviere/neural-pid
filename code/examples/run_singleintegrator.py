
from param import Param
from run import run
from systems.singleintegrator import SingleIntegrator
from other_policy import CBF

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple

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
		self.n_agents = 2
		self.r_comm = 2 #0.5
		self.r_obs_sense = 2.0
		self.r_agent = 0.2
		self.a_max = 0.5 
		self.a_min = -1*self.a_max
		self.r_safe = 2*self.r_agent 
		
		# 
		self.phi_max = self.a_max
		self.phi_min = -1*self.a_max
		
		# sim 
		self.sim_t0 = 0
		self.sim_tf = 100
		self.sim_dt = 0.1
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# IL
		self.il_train_model_fn = '../models/singleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/singleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.8
		self.il_batch_size = 100
		self.il_n_epoch = 5000
		self.il_lr = 5e-4
		self.il_n_data = 5000
		self.il_log_interval = 20
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Empty' # 'Empty','Barrier'
		self.controller_learning_module = 'DeepSet' # 

		# learning hyperparameters
		n,m,h = 4,2,128 # state dim, action dim, hidden layer
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(n,h),
			nn.Linear(h,h),
			nn.Linear(h,h)])

		self.il_phi_obs_network_architecture = nn.ModuleList([
			nn.Linear(2,h),
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

		self.max_neighbors = 3
		self.max_obstacles = 3

		# Sim
		self.sim_rl_model_fn = '../models/singleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/singleintegrator/il_current.pt'
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)

		# Barrier function stuff
		self.b_gamma = 0.1
		self.b_exph = 1 



if __name__ == '__main__':
	param = SingleIntegratorParam()
	env = SingleIntegrator(param)

	controllers = {
		'IL':	torch.load(param.sim_il_model_fn),
		# 'RL': torch.load(param.sim_rl_model_fn)
	}

	# if True:
	# 	if env.n_agents == 10:
	# 		# orca 10 ring
	# 		s0 = 0.8*np.array([50,0,0,0,40.4509,29.3893,0,0,15.4509,47.5528,0,0,-15.4509,47.5528,0,0,-40.4509,29.3893,0,0,-50,\
	# 			6.12323e-15,0,0,-40.4509,-29.3893,0,0,-15.4509,-47.5528,0,0,15.4509,-47.5528,0,0,40.4509,-29.3893,0,0])
	# 	elif env.n_agents == 20:
	# 		# orca 20 ring
	# 		s0 = 0.8*np.array([50,0,0,0,47.5528,15.4509,0,0,40.4509,29.3893,0,0,29.3893,40.4509,0,0,15.4509,47.5528,0,0,3.06162e-15,\
	# 			50,0,0,-15.4509,47.5528,0,0,-29.3893,40.4509,0,0,-40.4509,29.3893,0,0,-47.5528,15.4509,0,0,-50,6.12323e-15,\
	# 			0,0,-47.5528,-15.4509,0,0,-40.4509,-29.3893,0,0,-29.3893,-40.4509,0,0,-15.4509,-47.5528,0,0,-9.18485e-15,-50,\
	# 			0,0,15.4509,-47.5528,0,0,29.3893,-40.4509,0,0,40.4509,-29.3893,0,0,47.5528,-15.4509,0,0])
	# 	elif env.n_agents == 2:
	# 		# orca 2 line 
	# 		s0 = np.array([-4,0,0,0,4,0,0,0])
	# 	elif env.n_agents == 1:
	# 		# orca 1 
	# 		s0 = np.array([2,0,0,0])

	if True:
		InitialState = namedtuple('InitialState', ['start', 'goal'])

		# s0 = np.zeros((env.n))
		# r = 4.
		# d_rad = 2*np.pi/env.n_agents
		# for i in range(env.n_agents):
		# 	idx = env.agent_idx_to_state_idx(i) + \
		# 			np.arange(0,2)
			# s0[idx] = np.array([r*np.cos(d_rad*i),r*np.sin(d_rad*i)])
			# + 0.001*np.random.random(size=(1,2))
		# s0 = InitialState._make((s0, -s0))

		import yaml
		# with open("../baseline/centralized-planner/examples/test_2_agents.yaml") as map_file:
		# with open("../baseline/centralized-planner/examples/empty-8-8-random-1_30_agents.yaml") as map_file:
		with open("../baseline/centralized-planner/examples/map_8by8_obst12_agents10_ex0.yaml") as map_file:
			map_data = yaml.load(map_file)

		s = []
		g = []
		for agent in map_data["agents"]:
			s.extend([agent["start"][0] + 0.5, agent["start"][1] + 0.5])
			s.extend([0,0])
			g.extend([agent["goal"][0] + 0.5, agent["goal"][1] + 0.5])
			g.extend([0,0])

		InitialState = namedtuple('InitialState', ['start', 'goal'])
		s0 = InitialState._make((np.array(s), np.array(g)))

		env.obstacles = map_data["map"]["obstacles"]

	else:
		s0 = env.reset()

	run(param, env, controllers, s0)
