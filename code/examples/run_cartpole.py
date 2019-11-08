from param import Param
from run import run
from systems.cartpole import CartPole
import numpy as np
import torch
import torch.nn as nn 
from torch import tanh
import glob

class CartpoleParam(Param):
	def __init__(self):
		super().__init__()

		# env 
		self.env_name = 'CartPole'
		self.env_case = 'Any90' #'SmallAngle','Swing90','Swing180', 'Any90'

		# action constraints
		self.a_min = np.array([-10])
		self.a_max = np.array([10])

		# flags
		self.pomdp_on = False
		self.single_agent_sim = True
		self.multi_agent_sim = False

		# Sim
		self.sim_t0 = 0
		self.sim_tf = 10
		self.sim_dt = 0.05
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.sim_render_on = False
		self.sim_rl_model_fn = '../models/CartPole/rl_Any90_discrete.pt'
		self.sim_il_model_fn = '../models/CartPole/il_current.pt'
		self.sim_render_on = False

		# RL
		self.rl_train_model_fn = '../models/CartPole/rl_current.pt'
		self.rl_continuous_on = True
		self.rl_lr_schedule_on = False
		self.rl_lr_schedule = np.arange(0, self.sim_nt*1, 50)
		self.rl_lr_schedule_gamma = 0.2
		self.rl_gamma = 0.998
		self.rl_K_epoch = 10
		self.rl_batch_size = 1000
		self.rl_da = 2
		self.rl_discrete_action_space = np.arange(self.a_min, self.a_max, self.rl_da)
		self.rl_warm_start_on = False
		self.rl_warm_start_fn = '../models/CartPole/rl_current.pt'
		self.rl_module = 'DDPG' # 'DDPG','PPO','PPO_w_DeepSet'
		self.rl_scale_reward = 0.01 

		# dimensions
		n = 4 # state dim
		m = 1 # action dim
		h_mu = 32 # hidden layer
		h_q = 32 # hidden layer
		h_s = 32 # hidden layer discrete 

		if self.rl_continuous_on:
			# ddpg param
			self.rl_lr_mu = 1e-4
			self.rl_lr_q = 1e-3
			self.rl_buffer_limit = 1e6
			self.rl_action_std = 0.1
			self.rl_max_action_perturb = 1
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
			self.rl_lr = 1e-4
			self.rl_lmbda = 0.95
			self.rl_eps_clip = 0.2
			self.rl_layers = nn.ModuleList([
				nn.Linear(n,h_s),
				nn.Linear(h_s,h_s),
				nn.Linear(h_s,len(self.rl_discrete_action_space)),
				nn.Linear(h_s,1)
				])

		# IL
		h_i = 32 # hidden layers
		self.il_n_epoch = 50000 # number of epochs per batch
		self.il_batch_size = 500 # number of data points per batch
		self.il_n_data = 10000 # total number of data points
		self.il_lr = 2e-4
		self.il_log_interval = 100
		self.il_load_dataset = "../models/CartPole/dataset_rl/*.csv"
		self.il_load_dataset_on = True 
		self.il_test_train_ratio = 0.8
		self.il_state_loss_on = False
		self.il_train_model_fn = '../models/CartPole/il_current.pt'
		self.il_imitate_model_fn = '../models/CartPole/rl_current.pt'
		self.il_controller_class = 'Ref' # PID, PID_wRef, Ref, NL_EL
		self.il_K = np.eye(int(n/2))
		self.il_Lbda = np.eye(int(n/2))
		self.il_layers = nn.ModuleList([
			nn.Linear(n,h_i),
			nn.Linear(h_i,h_i),
			nn.Linear(h_i,h_i),
			nn.Linear(h_i,n)])
		self.il_activation = tanh
		self.il_kp = [2,4]
		self.il_kd = [0.3, 3.5]
		self.kp = [4.5,3]
		self.kd = [0.1, 0.5]

		# planning
		# self.rrt_fn = '../models/CartPole/rrt.csv'
		self.scp_fn = '../models/CartPole/scp.csv'
		self.scp_pdf_fn = '../models/CartPole/scp.pdf'

class PlainPID:
	"""
	Simple PID controller with fixed gains
	"""
	def __init__(self, Kp, Kd):
		self.Kp = Kp
		self.Kd = Kd

	def policy(self, state):
		action = (self.Kp[0]*state[0] + self.Kp[1]*state[1] + \
			self.Kd[0]*state[2] + self.Kd[1]*state[3])
		return action


class FilePolicy:
	def __init__(self, filename):
		data = np.loadtxt(filename, delimiter=',', ndmin=2)
		self.states = data[:,0:4]
		self.actions = data[:,4:5]
		self.steps = data.shape[0]


def find_best_file(path, x0):
	best_dist = None
	best_file = None
	best_x0 = None
	for file in glob.glob(path):
		data = np.loadtxt(file, delimiter=',', ndmin=2,max_rows=1)
		dist = np.linalg.norm(x0 - data[0,0:x0.shape[0]])
		if best_dist is None or dist < best_dist:
			best_dist = dist
			best_file = file
			best_x0 = data[0,0:x0.shape[0]]
	return best_file, best_x0


if __name__ == '__main__':
	param = CartpoleParam()
	env = CartPole(param)
	
	x0 = np.array([0.4, np.pi/2, 0.5, 0])
	# x0 = np.array([0.07438156, 0.33501733, 0.50978889, 0.52446423])

	# x0 = np.array([0,np.radians(180),0,0])

	# scp_file = find_best_file(param.il_load_dataset, x0)
	# print(scp_file)

	if param.il_load_dataset_on:
		scp_file, scp_x0 = find_best_file(param.il_load_dataset, x0)
		print(scp_file, scp_x0)

	controllers = {
		# 'RL':	torch.load('../models/CartPole/rl_Any90_discrete.pt'),
		# 'RL':	torch.load(param.sim_rl_model_fn),
		# 'IL':	torch.load(param.sim_il_model_fn),
		# 'PD': PlainPID(param.kp, param.kd),
		'SCP':	FilePolicy(scp_file),
	}


	run(param, env, controllers, x0)
