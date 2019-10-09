
from numpy import array,arange,zeros,Inf,linspace
import os,sys 

class Param:
	def __init__(self):

		# reinforcement learning parameters
		self.rl_gpu_on = False
		self.rl_continuous_on = True
		self.rl_log_interval = 20
		self.rl_save_model_interval = Inf
		self.rl_max_episodes = 50000
		self.rl_train_model_fn = 'rl_model.pt'
		self.rl_batch_size = 1000
		self.rl_gamma = 0.98
		self.rl_K_epoch = 5
		self.rl_control_lim = 10
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
		self.rl_lr_mu = 1e-4
		self.rl_lr_q = 1e-3
		self.rl_buffer_limit = 5e6
		self.rl_action_std = 2
		self.rl_max_action_perturb = 5
		self.rl_tau = 0.995

		# imitation learning parameters
		self.il_lr = 5e-4
		self.il_n_epoch = 50000 # number of epochs per batch 
		self.il_batch_size = 2000 # number of data points per batch
		self.il_n_data = 20000 # total number of data points 
		self.il_log_interval = 100
		self.il_train_model_fn = 'il_model.pt'
		self.il_imitate_model_fn = '../models/CartPole/rl_model_Swing90_continuous.pt'

		# dynamics (like openai env)
		self.env_name = 'MotionPlanner' # 'CartPole','MotionPlanner'
		self.env_case = None # Cartpole: 'SmallAngle','Swing90','Swing180', MotionPlanner: 'None'
		self.controller_class = 'Ref' # PID, PID_wRef, Ref

		# sim parameters
		self.sim_t0 = 0
		self.sim_tf = 10
		self.sim_dt = 0.05
		self.sim_times = arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.sim_rl_model_fn = 'rl_model.pt'
		self.sim_il_model_fn = 'il_model.pt'
		self.sim_render_on = False 

		# plots
		self.plots_fn = 'plots.pdf'

		# desired tracking trajectory
		self.ref_trajectory = zeros((4,self.sim_nt)) 

param = Param()


sys.path.insert(1, os.path.join(os.getcwd(),'learning'))
sys.path.insert(1, os.path.join(os.getcwd(),'systems'))