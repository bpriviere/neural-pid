
from numpy import array,arange,zeros,Inf,linspace


class Param:
	def __init__(self):

		# reinforcement learning parameters
		self.rl_gpu_on = True
		self.rl_continuous_on = True
		self.rl_log_interval = 20
		self.rl_save_model_interval = Inf
		self.rl_max_episodes = 50000
		self.rl_train_model_fn = 'rl_model.pt'
		self.rl_batch_size = 500
		self.rl_gamma = 0.98
		self.rl_K_epoch = 5
		self.rl_control_lim = 50
		self.rl_card_A = 25
		self.rl_discrete_action_space = linspace(\
			-self.rl_control_lim,
			 self.rl_control_lim,
			 self.rl_card_A)
		# ppo param
		self.rl_lr = 5e-3
		self.rl_lmbda = 0.95
		self.rl_eps_clip = 0.2
		# ddpg param
		self.rl_lr_mu = 5e-3
		self.rl_lr_q = 5e-3
		self.rl_buffer_limit = 5e6
		self.rl_action_std = 10
		self.rl_tau = 0.005

		# imitation learning parameters
		self.il_lr = 5e-4
		self.il_n_epoch = 1000 # number of epochs per batch 
		self.il_batch_size = 2000 # number of data points per batch
		self.il_n_data = 10000 # total number of data points 
		self.il_log_interval = 100
		self.il_train_model_fn = 'il_model.pt'
		self.il_imitate_model_fn = '../models/rl_model_SmallAngle_continuous.pt'

		# dynamics (like openai env)
		self.env_name = 'CartPole'
		self.env_case = 'Swing90' # 'SmallAngle','Swing90','Swing180'
		self.programmatic_controller_name = 'Ref' # PID, PID_wRef, Ref

		# sim parameters
		self.sim_t0 = 0
		self.sim_tf = 10
		self.sim_dt = 0.05
		self.sim_times = arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.sim_rl_model_fn = 'rl_model.pt'
		self.sim_il_model_fn = 'il_model.pt'

		# plots
		self.plots_fn = 'plots.pdf'

		# desired tracking trajectory
		self.ref_trajectory = zeros((4,self.sim_nt)) 

param = Param()