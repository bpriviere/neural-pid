
from param import Param
from run import run, parse_args
from sim import run_sim
from systems.singleintegrator_vel_sensing import SingleIntegratorVelSensing
from other_policy import APF, Empty_Net_wAPF
import plotter 

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple
import os

class SingleIntegratorVelSensingParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'SingleIntegratorVelSensing'
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
		self.r_agent = 0.15 #5
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
		self.phi_max = self.a_max + self.b_gamma/(0.2-self.r_agent) # 1*self.a_max
		self.phi_min = -self.phi_max # -1*self.a_max
		
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
		self.training_time_downsample = 100
		self.il_train_model_fn = '../models/singleintegrator_vel_sensing/il_current.pt'
		self.il_imitate_model_fn = '../models/singleintegrator_vel_sensing/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.85
		self.il_batch_size = 512 #5000
		self.il_n_epoch = 100
		self.il_lr = 1e-3
		self.il_wd = 0 #0.0002
		self.il_n_data = 500000 # 100000 # 100000000
		self.il_log_interval = 1
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Empty' # 'Empty','Barrier',
		
		self.datadict = dict()
		self.datadict["4"] = self.il_n_data
		self.datadict["20"] = self.il_n_data

		self.il_obst_case = 6
		self.controller_learning_module = 'DeepSet' #

		# adaptive dataset parameters
		self.adaptive_dataset_on = True
		self.ad_n = 100 # n number of rollouts
		self.ad_l = 2 # l prev observations 
		self.ad_k = 20 # k closest 
		self.ad_n_epoch = 10
		self.ad_n_data = 2000000
		self.ad_dl = 10 # every . timesteps  
		self.ad_train_model_fn = '../models/singleintegrator_vel_sensing/ad_current.pt'

		# Sim
		self.sim_rl_model_fn = '../models/singleintegrator_vel_sensing/rl_current.pt'
		self.sim_il_model_fn = '../models/singleintegrator_vel_sensing/il_current.pt'

		# plots
		self.vector_plot_dx = 0.25 		

		# self.ad_tf = 25 #25
		# self.ad_dt = 0.05
		# self.ad_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)

		# 
		self.il_empty_model_fn = '../models/singleintegrator_vel_sensing/empty.pt'
		self.il_barrier_model_fn = '../models/singleintegrator_vel_sensing/barrier.pt'
		self.il_adaptive_model_fn = '../models/singleintegrator_vel_sensing/adaptive.pt'

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
			nn.Linear(h,m)])

		self.il_network_activation = relu

if __name__ == '__main__':

	args = parse_args()
	if args.il:
		param = SingleIntegratorVelSensingParam()
		env = SingleIntegratorVelSensing(param)
		run(param, env, None, None, args)
		exit()

	set_ic_on = True

	if set_ic_on:

		import yaml
		if args.instance:
			with open(args.instance) as map_file:
				map_data = yaml.load(map_file,Loader=yaml.SafeLoader)
		else:
			# test map 
			ex = '0001' # 4 is hard 
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
		'IL':	torch.load(param.il_train_model_fn),
		'AD':	torch.load(param.ad_train_model_fn),
		# 'empty':	torch.load('../models/singleintegrator/empty.pt'),
		# 'barrier':	torch.load('../models/singleintegrator/barrier.pt'),
		# 'ILwAPF': Empty_Net_wAPF(param, env, torch.load(param.il_train_model_fn)),
		# 'ADwAPF': Empty_Net_wAPF(param, env, torch.load(param.ad_train_model_fn)),
		# 'APF': APF(param,env)
	}

	if args.batch:
				
		for name, controller in controllers.items():
			print("Running simulation with " + name)

			states, observations, actions, step = run_sim(param, env, controller, s0)
			states_and_actions = np.zeros((step, states.shape[1] + actions.shape[1]), dtype=states.dtype)
			states_and_actions[:,0::4] = states[:step,0::2]
			states_and_actions[:,1::4] = states[:step,1::2]
			states_and_actions[:,2::4] = actions[:step,0::2]
			states_and_actions[:,3::4] = actions[:step,1::2]

			result = np.hstack((param.sim_times[0:step].reshape(-1,1), states_and_actions))

			basename = os.path.splitext(os.path.basename(args.instance))[0]
			folder_name = "../results/singleintegrator/{}".format(name)
			if not os.path.exists(folder_name):
				os.mkdir(folder_name)
			output_file = "{}/{}.npy".format(folder_name, basename)
			with open(output_file, "wb") as f:
				np.save(f, result.astype(np.float32), allow_pickle=False)

	# elif args.export:
	# 	model = torch.load(param.il_train_model_fn)
	# 	model.export_to_onnx("IL")

	else:
		run(param, env, controllers, s0, args)
