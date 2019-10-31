
from param import Param
from run import run
from systems.consensus import Consensus
from other_policy import LCP_Policy, WMSR_Policy

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
		self.single_agent_sim = False
		self.multi_agent_sim = True 

		# RL
		self.rl_train_model_fn = '../models/consensus/rl_current.pt'
		self.rl_continuous_on = False
		self.rl_lr_schedule_on = False
		self.rl_gamma = 0.98
		self.rl_K_epoch = 5
		self.rl_da = 0.2 # action step size 
		self.a_min = -1
		self.a_max = 1
		self.rl_discrete_action_space = np.linspace(self.a_min, self.a_max, self.rl_da)
		self.rl_warm_start_on = False 
		self.rl_warm_start_fn = '../models/consensus/rl_current.pt'

		if self.rl_continuous_on:
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
			self.rl_network_activation = tanh  		
		else:
			# ppo param
			self.rl_lr = 5e-4
			self.rl_lmbda = 0.95
			self.rl_eps_clip = 0.2		

		# IL
		self.il_train_model_fn = '../models/consensus/il_current.pt'
		self.il_imitate_model_fn = '../models/consensus/rl_current.pt'
		self.il_controller_class = None 

		# Sim
		self.sim_rl_model_fn = '../models/consensus/rl_current.pt'
		self.sim_il_model_fn = '../models/consensus/il_current.pt'
		self.sim_render_on = True
		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.01
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)



if __name__ == '__main__':
	param = ConsensusParam()
	env = Consensus(param)

	# x0 = np.array([0.4, np.pi/2, 0.5, 0])
	# x0 = np.array([0.07438156, 0.33501733, 0.50978889, 0.52446423])


	controllers = {
		'LCP': LCP_Policy(env),
		# 'WMSR': WMSR_Policy(env),
		# 'RL':	torch.load('../models/CartPole/rl_Any90_discrete.pt'),
		# 'RL':	torch.load('../models/CartPole/rl_current.pt'),
		# 'IL':	torch.load(param.sim_il_model_fn),
		# 'SCP':	FilePolicy(scp_file),
	}

	run(param, env, controllers)