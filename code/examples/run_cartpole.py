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
		self.env_name = 'CartPole'
		self.env_case = 'Any90' #'SmallAngle','Swing90','Swing180', 'Any90'

		self.a_min = np.array([-10])
		self.a_max = np.array([10])

		# flags
		self.pomdp_on = False
		self.single_agent_sim = True
		self.multi_agent_sim = False

		# RL
		self.rl_train_model_fn = '../models/CartPole/rl_current.pt'
		self.rl_continuous_on = True
		self.rl_lr_schedule_on = False
		self.rl_gamma = 0.98
		self.rl_K_epoch = 5
		self.rl_discrete_action_space = np.linspace(self.a_min, self.a_max, 5)
		self.rl_warm_start_on = False 


		if self.rl_continuous_on:
			# ddpg param
			# ddpg param
			self.rl_lr_mu = 5e-5
			self.rl_lr_q = 5e-4
			self.rl_buffer_limit = 5e6
			self.rl_action_std = 1
			self.rl_max_action_perturb = 0.5
			self.rl_tau = 0.995
			# network architecture
			n,m,h_mu,h_q = 4,1,16,16 # state dim, action dim, hidden layers
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
			self.rl_lr = 2e-3
			self.rl_lmbda = 0.95
			self.rl_eps_clip = 0.2

		# IL
		self.il_lr = 1e-4
		self.il_load_dataset = "../models/CartPole/dataset_rl/*.csv"
		self.il_test_train_ratio = 0.8
		self.il_state_loss_on = False

		self.il_train_model_fn = '../models/CartPole/il_current.pt'
		self.il_imitate_model_fn = '../models/CartPole/rl_Any90_discrete.pt'
		self.kp = [2,4]
		self.kd = [0.3, 3.5]

		# Sim
		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.05
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)

		self.sim_rl_model_fn = '../models/CartPole/rl_current.pt'
		self.sim_il_model_fn = '../models/CartPole/il_current.pt'
		self.sim_render_on = False

		self.controller_class = 'PID_wRef' # PID, PID_wRef, Ref

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
	for file in glob.glob(path):
		data = np.loadtxt(file, delimiter=',', ndmin=2,max_rows=1)
		dist = np.linalg.norm(x0 - data[0,0:x0.shape[0]])
		if best_dist is None or dist < best_dist:
			best_dist = dist
			best_file = file
	return best_file


if __name__ == '__main__':
	param = CartpoleParam()
	env = CartPole(param)
	
	x0 = np.array([0.4, np.pi/2, 0.5, 0])

	scp_file = find_best_file(param.il_load_dataset, x0)
	print(scp_file)

	controllers = {
		'RL':	torch.load('../models/CartPole/rl_Any90_discrete.pt'),
		'IL':	torch.load(param.sim_il_model_fn),
		# 'PD': PlainPID(param.kp, param.kd),
		'SCP':	FilePolicy(scp_file),
	}


	run(param, env, controllers, x0)
