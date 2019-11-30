
# my package
from param import Param
from run import run
from systems.consensus import Consensus
from other_policy import LCP_Policy, WMSR_Policy

# standard package
import numpy as np 
from torch import nn as nn 
from torch import tanh
from torch.nn import LeakyReLU
import torch 

class ConsensusParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'Consensus'
		self.env_case = None

		#
		self.pomdp_on = True
		self.r_comm = 5
		self.n_agents = 5
		self.n_malicious = 1
		self.agent_memory = 3
		self.n_neighbors = 4
		self.single_agent_sim = False
		self.multi_agent_sim = True

		# Sim
		self.sim_rl_model_fn = '../models/consensus/rl_best.pt'
		self.sim_il_model_fn = '../models/consensus/il_current.pt'
		self.sim_render_on = True
		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.1
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# dim 
		self.state_dim_per_agent = 1 
		self.state_dim = self.n_agents*self.state_dim_per_agent
		self.action_dim_per_agent = 1
		self.action_dim = self.n_agents*self.action_dim_per_agent

		# env setup
		self.init_state_disturbance = 10*np.ones((self.state_dim),dtype=float)
		self.bias_disturbance = 10
		self.score_zone = 0.5
		self.env_state_bounds = 5*np.ones((self.state_dim),dtype=float)

		# RL
		self.rl_train_latest_model_fn = '../models/consensus/rl_latest.pt'
		self.rl_train_best_model_fn = '../models/consensus/rl_best.pt'
		self.rl_continuous_on = False
		self.rl_lr_schedule_on = True
		self.rl_lr_schedule = np.array([-np.inf, 0, 0.1, 0.2, 0.3])
		self.rl_lr_schedule_gamma = 0.5
		self.rl_gpu_on = False
		self.rl_max_episodes = 90000
		self.rl_batch_size = 1000
		self.rl_gamma = 0.99
		self.rl_K_epoch = 5
		self.rl_num_actions = 3
		self.a_min = -0.3
		self.a_max = 0.3
		self.rl_discrete_action_space = np.linspace(self.a_min, self.a_max, self.rl_num_actions)
		self.rl_warm_start_on = False 
		self.rl_warm_start_fn = '../models/consensus/rl_best.pt'
		self.rl_module = 'PPO' # PPO_w_DeepSet, DDPG, PPO, (DDPG_w_DeepSet)
		self.rl_log_interval = 20
		self.rl_scale_reward = 1.0

		h_s = 256 # hidden layer
		# self.rl_activation = LeakyReLU()
		self.rl_activation = tanh
		if self.rl_module is 'DDPG':
			# ddpg param
			self.rl_lr_mu = 5e-5
			self.rl_lr_q = 5e-4
			self.rl_buffer_limit = 5e6
			self.rl_action_std = 1
			self.rl_max_action_perturb = 0.5
			self.rl_tau = 0.995
			# network architecture
			self.rl_mu_network_architecture = nn.ModuleList([
				nn.Linear(n,h_mu),
				nn.Linear(h_mu,h_mu),
				nn.Linear(h_mu,m)])
			self.rl_q_network_architecture = nn.ModuleList([
				nn.Linear(n+m,h_q),
				nn.Linear(h_q,h_q),
				nn.Linear(h_q,1)])

		elif self.rl_module is 'PPO':
			# ppo param
			self.rl_lr = 1e-3
			self.rl_lmbda = 0.95
			self.rl_eps_clip = 0.2

			# self.rl_layers = nn.ModuleList([
			# 	nn.Linear(self.agent_memory*self.n_neighbors,h_s),
			# 	nn.Linear(h_s,len(self.rl_discrete_action_space)),
			# 	nn.Linear(h_s,1)
			# 	])

			self.rl_pi_layers = nn.ModuleList([
				nn.Linear(self.agent_memory*self.state_dim_per_agent*(self.n_neighbors+1),h_s),
				nn.Linear(h_s,h_s),
				nn.Linear(h_s,h_s),
				nn.Linear(h_s,len(self.rl_discrete_action_space))
				])

			self.rl_v_layers = nn.ModuleList([
				nn.Linear(self.agent_memory*self.state_dim_per_agent*(self.n_neighbors+1),h_s),
				nn.Linear(h_s,h_s),
				nn.Linear(h_s,h_s),
				nn.Linear(h_s,self.action_dim_per_agent)
				])

		# IL
		self.il_train_model_fn = '../models/consensus/il_current.pt'
		self.il_imitate_model_fn = '../models/consensus/rl_best.pt'
		self.il_controller_class = 'Consensus'
		self.il_module = 'DeepSet'
		self.il_load_dataset_on = True 
		self.il_test_train_ratio = 0.8
		self.il_n_data = 10000
		self.il_batch_size = 500 
		self.il_n_epoch = 50000
		self.il_lr = 1e-4
		self.il_log_interval = 2
		self.il_state_loss_on = False
		self.il_load_dataset = '../data/consensus/'
		self.il_bw_threshold = .02

		h_phi = 128
		h_rho = h_phi + self.state_dim_per_agent*self.agent_memory
		self.il_network_activation = tanh
		if self.il_module is 'DeepSet':
			
			self.il_phi_network_architecture = nn.ModuleList([
				nn.Linear(self.state_dim_per_agent*self.agent_memory,h_phi),
				nn.Linear(h_phi,h_phi),
				nn.Linear(h_phi,h_phi)
				])
			self.il_rho_network_architecture = nn.ModuleList([
				nn.Linear(h_rho,h_rho),
				nn.Linear(h_rho,h_rho),
				nn.Linear(h_rho,self.action_dim_per_agent*self.n_neighbors)])


if __name__ == '__main__':
	param = ConsensusParam()
	env = Consensus(param)

	# x0 = np.array([0.4, np.pi/2, 0.5, 0])
	# x0 = np.array([0.07438156, 0.33501733, 0.50978889, 0.52446423])


	x0 = env.reset()
	while not env.bad_nodes[2]:
		x0 = env.reset()

	controllers = {
		'LCP': LCP_Policy(env),
		'WMSR': WMSR_Policy(env),
		'RL':	torch.load(param.sim_rl_model_fn),
		'IL':	torch.load(param.sim_il_model_fn),
	}

	run(param, env, controllers, initial_state = x0)