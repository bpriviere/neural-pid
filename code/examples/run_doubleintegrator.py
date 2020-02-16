
from param import Param
from run import run, parse_args
from sim import run_sim
from systems.doubleintegrator import DoubleIntegrator
from other_policy import APF, Empty_Net_wAPF, ZeroPolicy, GoToGoalPolicy
import plotter 

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple
import os

class DoubleIntegratorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'DoubleIntegrator'
		self.env_case = None

		# flags
		self.pomdp_on = True
		self.single_agent_sim = False
		self.multi_agent_sim = True
		self.il_state_loss_on = False
		self.sim_render_on = False

		# orca param
		self.n_agents = 4
		self.r_comm = 3. #0.5
		self.r_obs_sense = 3.
		self.r_agent = 0.15 #0.2
		self.r_obstacle = 0.5
		self.v_max = 0.5
		self.a_max = 2.0
		# self.v_max = np.inf
		# self.a_max = np.inf 
		self.v_min = -1*self.v_max
		self.a_min = -1*self.a_max

		# sim 
		self.sim_t0 = 0
		self.sim_tf = 30
		self.sim_dt = 0.05
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# safety
		self.Delta_R = 2*(self.v_max*self.sim_dt + 1/2*self.a_max*self.sim_dt**2) + 1e-6

		self.max_neighbors = 6
		self.max_obstacles = 6
		
		# cbf 
		self.cbf_kp = 2.0
		self.cbf_kd = 4.0
		
		self.pi_max = 2.0 #0.5 #0.5 #0.10 #1.0*self.a_max
		self.sigmoid_scale = 1.0
		
		self.safety = "cf_di" # potential, fdbk_di, cf_di
		self.rollout_batch_on = True
		self.kp = 0.05
		self.kv = 0.05

		# obsolete parameters 
		self.b_gamma = .05 
		self.b_eps = 100
		self.b_exph = 1.0 
		self.D_robot = 1.*(self.r_agent+self.r_agent)
		self.D_obstacle = 1.*(self.r_agent + self.r_obstacle)
		self.circle_obstacles_on = True # square obstacles batch not implemented		

		# IL
		self.il_load_loader_on = True
		self.training_time_downsample = 50 #10
		self.il_train_model_fn = '../models/doubleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/doubleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.85
		self.il_batch_size = 4096*8
		self.il_n_epoch = 200
		self.il_lr = 1e-3
		self.il_wd = 0 #0.0002
		self.il_n_data = None # 100000 # 100000000
		self.il_log_interval = 1
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Barrier' # 'Empty','Barrier',
		self.il_pretrain_weights_fn = None # None or path to *.tar file
		
		self.datadict = dict()
		# self.datadict["4"] = 10000 #self.il_n_data
		self.datadict["obst"] = 7000000 #10000000 #750000 #self.il_n_data
		# self.datadict["10"] = 10000000 #250000 #self.il_n_data
		# self.datadict["15"] = 10000000 #250000 #self.il_n_data
		# self.datadict["012"] = 1000000 #250000 #self.il_n_data
		# self.datadict["032"] = 1000000 #250000 #self.il_n_data

		self.il_obst_case = 12
		self.controller_learning_module = 'DeepSet' #

		# adaptive dataset parameters
		self.adaptive_dataset_on = False
		self.ad_n = 100 # n number of rollouts
		self.ad_n_data_per_rollout = 100000 # repeat rollout until at least this amount of data was added
		self.ad_l = 2 # l prev observations 
		self.ad_k = 20 # k closest 
		self.ad_n_epoch = 10
		self.ad_n_data = 2000000
		self.ad_dl = 10 # every . timesteps  
		self.ad_train_model_fn = '../models/doubleintegrator/ad_current.pt'

		# Sim
		self.sim_rl_model_fn = '../models/doubleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/doubleintegrator/il_current.pt'

		# plots
		self.vector_plot_dx = 0.25 		

		# self.ad_tf = 25 #25
		# self.ad_dt = 0.05
		# self.ad_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)

		# 
		# self.il_empty_model_fn = '../models/singleintegrator/empty.pt'
		# self.il_barrier_model_fn = '../models/singleintegrator/barrier.pt'
		# self.il_adaptive_model_fn = '../models/singleintegrator/adaptive.pt'

		# learning hyperparameters
		n,m,h,l,p = 4,2,64,16,16 # state dim, action dim, hidden layer, output phi, output rho
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(4,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_phi_obs_network_architecture = nn.ModuleList([
			nn.Linear(4,h),
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
			nn.Linear(2*p+4,h),
			nn.Linear(h,h),
			nn.Linear(h,m)])

		self.il_network_activation = relu

		# plots
		self.vector_plot_dx = 0.3


def load_instance(param, env, instance):
	import yaml
	if instance:
		with open(instance) as map_file:
			map_data = yaml.load(map_file,Loader=yaml.SafeLoader)
	else:
		# default
		# instance = "map_8by8_obst6_agents64_ex0006.yaml"
		# instance = "map_8by8_obst6_agents32_ex0005.yaml"
		# instance = "map_8by8_obst6_agents16_ex0003.yaml"
		instance = "map_8by8_obst6_agents4_ex0007.yaml"
		# instance = "head_test.yaml"
		# with open("../results/singleintegrator/instances/{}".format(instance)) as map_file:
		with open("../results/singleintegrator/instances/{}".format(instance)) as map_file:
		# test map test dataset
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

	param.n_agents = len(map_data["agents"])
	env.reset_param(param)

	env.obstacles = map_data["map"]["obstacles"]
	for x in range(-1,map_data["map"]["dimensions"][0]+1):
		env.obstacles.append([x,-1])
		env.obstacles.append([x,map_data["map"]["dimensions"][1]])
	for y in range(map_data["map"]["dimensions"][0]):
		env.obstacles.append([-1,y])
		env.obstacles.append([map_data["map"]["dimensions"][0],y])

	return s0


def run_batch(param, env, instance, controllers):
	torch.set_num_threads(1)
	s0 = load_instance(param, env, instance)
	for name, controller in controllers.items():
		print("Running simulation with " + name)

		states, observations, actions, step = run_sim(param, env, controller, s0)
		# print(states[0:step].shape)
		# print(param.sim_times[0:step].shape)
		# exit()
		result = np.hstack((param.sim_times[0:step].reshape(-1,1), states[0:step]))
		# store in binary format
		basename = os.path.splitext(os.path.basename(instance))[0]
		folder_name = "../results/doubleintegrator/{}".format(name)
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)

		output_file = "{}/{}.npy".format(folder_name, basename)
		with open(output_file, "wb") as f:
			np.save(f, result.astype(np.float32), allow_pickle=False)

if __name__ == '__main__':

	args = parse_args()
	param = DoubleIntegratorParam()
	env = DoubleIntegrator(param)

	if args.il:
		run(param, env, None, None, args)
		exit()

	controllers = {
		# 'emptywapf': Empty_Net_wAPF(param,env,torch.load('../results/doubleintegrator/exp1Empty_0/il_current.pt')),
		# 'barrier':torch.load('../results/doubleintegrator/exp1Barrier_0/il_current.pt'),
		# 'empty':torch.load('../results/doubleintegrator/exp1Empty_0/il_current.pt'),
		# 
		# 'current':torch.load(param.il_train_model_fn),
		# 'current_wapf': Empty_Net_wAPF(param,env,torch.load(param.il_train_model_fn)),
		# 'gg': GoToGoalPolicy(param,env),
		'apf': Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env)),
		# 'zero': Empty_Net_wAPF(param,env,ZeroPolicy(env))
	}

	s0 = load_instance(param, env, args.instance)

	if args.batch:
		if args.controller:
			controllers = dict()
			for ctrl in args.controller:
				name,kind,path = ctrl.split(',')
				if kind == "EmptyAPF":
					controllers[name] = Empty_Net_wAPF(param,env,torch.load(path))
				elif kind == "torch":
					controllers[name] = torch.load(path)
				elif kind == "apf":
					controllers[name] = Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env))
				else:
					print("ERROR unknown ctrl kind", kind)
					exit()

		if args.Rsense:
			param.r_comm = args.Rsense
			param.r_obs_sense = args.Rsense
		if args.maxNeighbors:
			param.max_neighbors = args.maxNeighbors
			param.max_obstacles = args.maxNeighbors
		env.reset_param(param)

		run_batch(param, env, args.instance, controllers)

	elif args.export:
		# model = torch.load('/home/whoenig/pCloudDrive/caltech/neural_pid_results/doubleintegrator/il_current.pt')
		# change path 
		model = torch.load('/home/ben/pCloudDrive/arcl/neural_pid/results/neural_pid_results/doubleintegrator/il_current.pt')
		model.export_to_onnx("IL")

	else:
		run(param, env, controllers, s0, args)

