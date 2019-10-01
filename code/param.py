
from numpy import array,arange,zeros,Inf


class Param:
	def __init__(self):

		# reinforcement learning parameters
		self.rl_action_std = 1.5 
		self.rl_cuda_on = False
		self.rl_continuous_on = False
		self.rl_log_interval = 20
		self.rl_save_model_interval = Inf
		self.rl_lr = 5e-3
		self.rl_gamma = 0.98 
		self.rl_lmbda = 0.95
		self.rl_eps_clip = 0.2
		self.rl_K_epoch = 5
		self.rl_max_episodes = 50000
		self.rl_model_fn = 'rl_model_swing180_discrete_working.pt'
		self.rl_ndata_per_epi = 1000

		# imitation learning parameters
		self.il_lr = 5e-4
		self.il_n_epoch = 500
		self.il_batch_size = 250
		self.il_n_data = 10000
		self.il_log_interval = 100
		self.il_model_fn = 'il_model.pt'

		# dynamics (like openai env)
		self.env_name = 'CartPole'
		self.env_case = 'Swing_180' # 'SmallAngle','Swing_90','Swing_180'

		# sim parameters
		self.sim_t0 = 0
		self.sim_tf = 10
		self.sim_dt = 0.05
		self.sim_times = arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)

		# plots
		self.plots_fn = 'plots.pdf'

		# desired tracking trajectory
		self.ref_trajectory = zeros((4,self.sim_nt)) 

param = Param()