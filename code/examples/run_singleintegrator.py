
from param import Param
from run import run, parse_args
from sim import run_sim
from systems.singleintegrator import SingleIntegrator
from other_policy import APF, Empty_Net_wAPF
import plotter 

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple
import os

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
		self.n_agents = 1
		self.r_comm = 3. #0.5
		self.r_obs_sense = 3.
		self.r_agent = 0.22 #5
		self.r_obstacle = 0.5
		self.a_max = 0.5
		self.a_min = -1*self.a_max
		self.D_robot = 1.*(self.r_agent+self.r_agent)
		self.D_obstacle = 1.*(self.r_agent + self.r_obstacle)
		self.circle_obstacles_on = True # square obstacles batch not implemented

		self.max_neighbors = 5
		self.max_obstacles = 5
		# Barrier function stuff
		self.b_gamma = 0.05 # 0.1
		self.b_exph = 1.0 # 1.0
		# cbf 
		# self.cbf_kp = 1.0
		# self.cbf_kv = 0.1
		# self.a_noise = 0.002

		# 
		self.phi_max = 1*self.a_max
		self.phi_min = -1*self.a_max
		
		# sim 
		self.sim_t0 = 0
		self.sim_tf = 50 #25
		self.sim_dt = 0.05
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# IL
		# self.il_load_loader_on = True
		self.il_load_loader_on = False
		self.training_time_downsample = 100 #10
		self.il_train_model_fn = '../models/singleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/singleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.85
		self.il_batch_size = 512 # 512 #5000
		self.il_n_epoch = 500
		self.il_lr = 1e-3
		self.il_wd = 0 #0.0002
		self.il_n_data = 100000000 # 100000 # 100000000
		self.il_log_interval = 1
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Barrier' # 'Empty','Barrier',
		self.controller_learning_module = 'DeepSet' #

		# 
		self.il_empty_model_fn = '../models/singleintegrator/empty.pt'
		self.il_barrier_model_fn = '../models/singleintegrator/barrier.pt'

		# learning hyperparameters
		n,m,h,l,p = 2,2,32,8,8 # state dim, action dim, hidden layer, output phi, output rho
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(2,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_phi_obs_network_architecture = nn.ModuleList([
			nn.Linear(2,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_rho_network_architecture = nn.ModuleList([
			nn.Linear(l,h),
			nn.Linear(h,h),
			nn.Linear(h,p)])

		self.il_rho_obs_network_architecture = nn.ModuleList([
			nn.Linear(l,h),
			nn.Linear(h,h),
			nn.Linear(h,p)])

		self.il_psi_network_architecture = nn.ModuleList([
			nn.Linear(2*p+2,h),
			nn.Linear(h,h),
			nn.Linear(h,h),
			nn.Linear(h,m)])

		self.il_network_activation = relu

		# Sim
		self.sim_rl_model_fn = '../models/singleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/singleintegrator/il_current.pt'
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)


if __name__ == '__main__':

	args = parse_args()
	if args.il:
		param = SingleIntegratorParam()
		env = SingleIntegrator(param)
		run(param, env, None, None, args)
		exit()

	set_ic_on = True
	ring_ex_on = False


	# TEMP
	# param = SingleIntegratorParam()
	# env = SingleIntegrator(param)			
	# plotter.plot_barrier_fnc(env)
	# plotter.save_figs(param.plots_fn)
	# plotter.open_figs(param.plots_fn)
	# exit()

	if set_ic_on:

		if ring_ex_on:

			param = SingleIntegratorParam()
			env = SingleIntegrator(param)			

			InitialState = namedtuple('InitialState', ['start', 'goal'])

			s0 = np.zeros((env.n))
			r = 4.
			d_rad = 2*np.pi/env.n_agents
			for i in range(env.n_agents):
				idx = env.agent_idx_to_state_idx(i) + \
						np.arange(0,2)
				s0[idx] = np.array([r*np.cos(d_rad*i),r*np.sin(d_rad*i)])
				+ 0.001*np.random.random(size=(1,2))
			s0 = InitialState._make((s0, -s0))

		else:

			import yaml
			ex = '0001' # 4 is hard 
			
			if args.instance:
				with open(args.instance) as map_file:
					map_data = yaml.load(map_file)
			else:
				# test map 
				with open("../results/singleintegrator/instances/map_8by8_obst6_agents4_ex{}.yaml".format(ex)) as map_file:

				# test map test dataset

					map_data = yaml.load(map_file)

			s = []
			g = []
			for agent in map_data["agents"]:
				s.extend([agent["start"][0] + 0.5, agent["start"][1] + 0.5])
				g.extend([agent["goal"][0] + 0.5, agent["goal"][1] + 0.5])

			InitialState = namedtuple('InitialState', ['start', 'goal'])
			s0 = InitialState._make((np.array(s), np.array(g)))

			param = SingleIntegratorParam()
			param.n_agents = len(map_data["agents"])
			env = SingleIntegrator(param)

			env.obstacles = map_data["map"]["obstacles"]
			for x in range(-1,map_data["map"]["dimensions"][0]+1):
				env.obstacles.append([x,-1])
				env.obstacles.append([x,map_data["map"]["dimensions"][1]])
			for y in range(map_data["map"]["dimensions"][0]):
				env.obstacles.append([-1,y])
				env.obstacles.append([map_data["map"]["dimensions"][0],y])

	else:
		s0 = env.reset()

	controllers = {
		# 'IL':	torch.load(param.sim_il_model_fn),
		# 'APF': APF(param,env)
		# 'empty': torch.load(param.il_empty_model_fn),
		'empty_wAPF' : Empty_Net_wAPF(param,env),
		'barrier' : torch.load(param.il_barrier_model_fn)
	}

	if args.batch:
		for name, controller in controllers.items():
			print("Running simulation with " + name)
			states, observations, actions, step = run_sim(param, env, controller, s0)
			states_and_actions = np.zeros((actions.shape[0], states.shape[1] + actions.shape[1]), dtype=states.dtype)
			states_and_actions[:,0::4] = states[:-1,0::2]
			states_and_actions[:,1::4] = states[:-1,1::2]
			states_and_actions[:,2::4] = actions[:,0::2]
			states_and_actions[:,3::4] = actions[:,1::2]

			result = np.hstack((param.sim_times[0:-1].reshape(-1,1), states_and_actions))
			# store in binary format
			basename = os.path.splitext(os.path.basename(args.instance))[0]
			folder_name = "../results/singleintegrator/{}".format(name)
			if not os.path.exists(folder_name):
				os.mkdir(folder_name)
			output_file = "{}/{}.npy".format(folder_name, basename)
			with open(output_file, "wb") as f:
				np.save(f, result, allow_pickle=False)
	else:
		run(param, env, controllers, s0, args)

