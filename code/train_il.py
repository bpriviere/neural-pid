
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np 
import random 
import glob
import os
import yaml
import utilities

from numpy import array, zeros, Inf
from numpy.random import uniform,seed
from torch.distributions import Categorical
from collections import namedtuple
from torch.optim.lr_scheduler import ReduceLROnPlateau

from learning.pid_net import PID_Net
from learning.pid_wref_net import PID_wRef_Net
from learning.ref_net import Ref_Net
from learning.empty_net import Empty_Net
from learning.barrier_net import Barrier_Net
from learning.nl_el_net import NL_EL_Net
from learning.consensus_net import Consensus_Net


def load_orca_dataset_action_loss(filename,neighborDist,obstacleDist,max_neighbors,max_obstacles,training_time_downsample):
	data = np.load(filename)
	data = torch.from_numpy(data)

	# load map
	instance = os.path.splitext(os.path.basename(filename))[0]

	filename_map = "{}/../instances/{}.yaml".format(os.path.dirname(filename), instance)
	with open(filename_map) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)
	obstacles = []
	for o in map_data["map"]["obstacles"]:
		obstacles.append(torch.Tensor(o) + torch.Tensor([0.5,0.5]))

	for x in range(-1,map_data["map"]["dimensions"][0]+1):
		obstacles.append(torch.Tensor([x,-1]) + torch.Tensor([0.5,0.5]))
		obstacles.append(torch.Tensor([x,map_data["map"]["dimensions"][1]]) + torch.Tensor([0.5,0.5]))
	for y in range(map_data["map"]["dimensions"][0]):
		obstacles.append(torch.Tensor([-1,y]) + torch.Tensor([0.5,0.5]))
		obstacles.append(torch.Tensor([map_data["map"]["dimensions"][0],y]) + torch.Tensor([0.5,0.5]))


	num_agents = int((data.shape[1] - 1) / 4)
	dataset = []
	# Observation_Action_Pair = namedtuple('Observation_Action_Pair', ['observation', 'action']) 
	# Observation = namedtuple('Observation',['relative_goal','time_to_goal','relative_neighbors','relative_obstacles']) 
	for t in range(data.shape[0]-1):
		if t%training_time_downsample != 0:
			continue
		for i in range(num_agents):
			s_i = data[t,i*4+1:i*4+5]   # state i 
			# s_g = data[-1,i*4+1:i*4+5]  # goal state i 
			s_g = torch.Tensor(map_data["agents"][i]["goal"] + [0,0]) + torch.Tensor([0.5,0.5,0,0])
			# print(s_g, data[-1,i*4+1:i*4+5])
			relative_goal = s_g - s_i   # relative goal
			if np.allclose(relative_goal, np.zeros(4)):
				continue
			time_to_goal = data[-1,0] - data[t,0]
			relative_neighbors = []
			for j in range(num_agents):
				if i != j:
					s_j = data[t,j*4+1:j*4+5] # state j
					# dist = np.linalg.norm(s_i[0:2] - s_j[0:2])
					dist = (s_j[0:2] - s_i[0:2]).norm()
					if dist <= neighborDist:
						relative_neighbors.append(s_j - s_i)
						# print(dist, len(relative_neighbors))
						# break
			relative_neighbors.sort(key=lambda n: n[0:2].norm())
			del relative_neighbors[max_neighbors:]

			relative_obstacles = []
			for o in obstacles:
				dist = (o - s_i[0:2]).norm()
				if dist <= obstacleDist:
					relative_obstacles.append(o - s_i[0:2])
			relative_obstacles.sort(key=lambda o: o.norm())
			del relative_obstacles[max_obstacles:]

			num_neighbors = len(relative_neighbors)
			num_obstacles = len(relative_obstacles)

			obs_array = np.empty(5+4*num_neighbors+2*num_obstacles+2, dtype=np.float32)
			obs_array[0] = num_neighbors
			idx = 1
			obs_array[idx:idx+4] = relative_goal
			idx += 4
			# obs_array[4] = data.observation.time_to_goal
			for k in range(num_neighbors):
				obs_array[idx:idx+4] = relative_neighbors[k]
				idx += 4
			for k in range(num_obstacles):
				obs_array[idx:idx+2] = relative_obstacles[k]
				idx += 2
			obs_array[idx:idx+2] = data[t+1, i*4+3:i*4+5]
			idx += 2

			dataset.append(obs_array)

			# o = Observation._make((
			# 	relative_goal,
			# 	time_to_goal,
			# 	relative_neighbors,
			# 	relative_obstacles))
			# # a = data[t+1, i*4+3:i*4+5].clone().detach().numpy() # desired control is the velocity in the next timestep
			# a = np.array(data[t+1, i*4+3:i*4+5], dtype=np.float32)
			# oa_pair = Observation_Action_Pair._make((o,a))
			# dataset.append(oa_pair)
			# break
	print('Dataset Size: ',len(dataset))

	# import plotter
	# from matplotlib.patches import Rectangle
	# robot = 0
	# for item in dataset:
	# 	fig,ax = plotter.make_fig()
	# 	ax.set_title('State')
	# 	ax.set_aspect('equal')

	# 	ax.set_xlim([-1,10])
	# 	ax.set_ylim([-1,10])

	# 	# plot all obstacles
	# 	for o in obstacles:
	# 		ax.add_patch(Rectangle(o - torch.Tensor([0.5,0.5]), 1.0, 1.0, facecolor='gray', alpha=0.5))

	# 	# plot current position
	# 	s_g = data[-1,robot*4+1:robot*4+5]
	# 	robot_pos = s_g - item.observation.relative_goal
	# 	plotter.plot_circle(robot_pos[0], robot_pos[1],0.2,fig=fig,ax=ax)

	# 	# plot current observation
	# 	for i, obs in enumerate(item.observation.relative_obstacles):
	# 		pos = obs + robot_pos[0:2] - torch.Tensor([0.5,0.5])
	# 		ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
	# 		if i >= max_obstacles-1:
	# 			break

	# plotter.save_figs(filename + ".pdf")
	# plotter.open_figs(filename + ".pdf")

	return dataset

def load_consensus_dataset(filename,n_neighbor,agent_memory):
	
	dataset = []
	data = np.load(filename)
	Observation_Action_Pair = namedtuple('Observation_Action_Pair', ['observation', 'action']) 

	for t in range(1,data.shape[0]-1):
		relative_neighbor_histories = []

		self_history = []
		for h in range(agent_memory):
			self_history.append(data[t,h])
		relative_neighbor_histories.append(self_history)

		for i in range(n_neighbor):
			relative_neighbor_history = []

			for h in range(agent_memory):
				relative_neighbor_history.append(data[t,(i+1)*agent_memory+h])
			relative_neighbor_histories.append(relative_neighbor_history)

		o = relative_neighbor_histories
		a = [data[t,-1]]

		oa_pair = Observation_Action_Pair._make((o,a))
		dataset.append(oa_pair)
	print('Dataset Size: ', len(dataset))
	return dataset

def make_loader(
	dataset=None,
	n_data=None,
	shuffle=False,
	batch_size=None,
	preprocess_transformation=True,
	name=None):

	def batch_loader(dataset):
		# break by observation size
		dataset_dict = dict()

		for data in dataset:
			num_neighbors = int(data[0])
			num_obstacles = int((data.shape[0] - num_neighbors*4 - 5 - 2) / 2)
			key = (num_neighbors, num_obstacles)
			if key in dataset_dict:
				dataset_dict[key].append(data)
			else:
				dataset_dict[key] = [data]

		# Create actual batches
		loader = []
		for key, dataset_per_key in dataset_dict.items():
			num_neighbors, num_obstacles = key
			batch_x = []
			batch_y = []
			for data in dataset_per_key:
				batch_x.append(data[0:-2])
				batch_y.append(data[-2:])

			# store all the data for this nn/no-pair in a file
			batch_x = np.array(batch_x)
			batch_y = np.array(batch_y)

			print(name, " neighbors ", num_neighbors, " obstacles ", num_obstacles, " ex. ", batch_x.shape[0])

			with open("../preprocessed_data/batch_{}_nn{}_no{}.npy".format(name,num_neighbors, num_obstacles), "wb") as f:
				np.save(f, np.hstack((batch_x, batch_y)), allow_pickle=False)

			# split data by batch size
			for idx in np.arange(0, batch_x.shape[0], batch_size):
				last_idx = min(idx + batch_size, batch_x.shape[0])
				# print("Batch of size ", last_idx - idx)
				x_data = torch.from_numpy(batch_x[idx:last_idx]).float()
				y_data = torch.from_numpy(batch_y[idx:last_idx]).float()
				loader.append([x_data, y_data])

		return loader


	if dataset is None:
		raise Exception('dataset not specified')
	
	if shuffle:
		random.shuffle(dataset)

	if n_data is not None and n_data < len(dataset):
		dataset = dataset[0:n_data]

	loader = batch_loader(dataset)

	if preprocess_transformation:
		loader_numpy,_ = utilities.preprocess_transformation(loader)
		loader = [(torch.Tensor(o),torch.Tensor(a)) for o,a in loader_numpy]
	return loader

def load_loader(name,batch_size,preprocess_transformation):

	loader = []
	datadir = glob.glob("../preprocessed_data/batch_{}*.npy".format(name))
	for file in datadir: 
		
		batch = np.load(file)
		batch_x = batch[:,0:-2]
		batch_y = batch[:,-2:]

		# split data by batch size
		for idx in np.arange(0, batch_x.shape[0], batch_size):
			last_idx = min(idx + batch_size, batch_x.shape[0])
			# print("Batch of size ", last_idx - idx)
			x_data = torch.from_numpy(batch_x[idx:last_idx]).float()
			y_data = torch.from_numpy(batch_y[idx:last_idx]).float()
			loader.append([x_data, y_data])

	if preprocess_transformation:
		loader_numpy,_ = utilities.preprocess_transformation(loader)
		loader = [(torch.Tensor(o),torch.Tensor(a)) for o,a in loader_numpy]
	return loader


def train(param,env,model,optimizer,loader):

	
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	# loss_func = torch.nn.L1Loss()
	epoch_loss = 0

	for step, (b_x, b_y) in enumerate(loader): # for each training step

		prediction = model(b_x)     # input x and predict based on x
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # apply gradients
		
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def test(param,env,model,loader):

	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	# loss_func = torch.nn.L1Loss()  
	epoch_loss = 0

	for step, (b_x, b_y) in enumerate(loader): # for each training step

		# # convert b_y if necessary
		# if not isinstance(b_y, torch.Tensor):
		# 	b_y = torch.from_numpy(np.array(b_y)).float()

		prediction = model(b_x)     # input batch state and predict batch action

		# if param.il_state_loss_on:
		# 	prediction_a = prediction
		# 	prediction = torch.zeros((b_y.shape))
		# 	for k,a in enumerate(prediction_a): 
		# 		prediction[k,:] = env.next_state_training_state_loss(b_x[k],a)

		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def train_il(param, env):

	seed(1) # numpy random gen seed 
	torch.manual_seed(1)    # pytorch 

	# init model
	if param.il_controller_class is 'NL_EL':
		model = NL_EL_Net(env, param.il_K, param.il_Lbda, param.il_layers, param.il_activation)
	elif param.il_controller_class is 'PID':
		model = PID_Net(env.n, env.m)
	elif param.il_controller_class is 'PID_wRef':
		model = PID_wRef_Net(env.m, param.il_layers, param.il_activation)
	elif param.il_controller_class is 'Ref':
		model = Ref_Net(env.n, env.m, env.a_min, env.a_max, param.kp, param.kd, param.il_layers, param.il_activation)
	elif param.il_controller_class is 'Barrier':
		model = Barrier_Net(param,param.controller_learning_module)
	elif param.il_controller_class is 'Consensus':
		model = Consensus_Net(param,param.il_module)
	elif param.il_controller_class is 'Empty':
		model = Empty_Net(param,param.controller_learning_module) 
	else:
		print('Error in Train Gains, programmatic controller not recognized')
		exit()

	print("Case: ",param.env_case)
	print("Controller: ",param.il_controller_class)

	# datasets
	if param.il_load_dataset_on:

		# orca dataset
		if "orca" in param.il_load_dataset:

			if "ring" in param.il_load_dataset:
				datadir = glob.glob("../data/singleintegrator/ring/*.npy")
			elif "random" in param.il_load_dataset:
				datadir = glob.glob("../data/singleintegrator/random/*.npy")
			elif "centralplanner" in param.il_load_dataset:
				
				# 1 agent cases
				# datadir = glob.glob("../data/singleintegrator/central/*agents1_*")

				# 10 agent cases
				# datadir = glob.glob("../data/singleintegrator/central/*agents10*")
				
				# primitive cases
				# datadir = glob.glob("../data/singleintegrator/central/*primitive*")

				# 6 obst cases
				datadir = glob.glob("../data/singleintegrator/central/*obst6_agents4_ex*.npy")
				# datadir.extend(glob.glob("../data/singleintegrator/central/*obst12_agents1_*"))

				# single case ex (to overfit)
				# datadir = glob.glob("../data/singleintegrator/central_single_case_2/*.npy")

				# 1 agent cases and primitives
				# datadir = glob.glob("../data/singleintegrator/central/*agents1_*")
				# datadir.extend(glob.glob("../data/singleintegrator/central/*primitive*"))

			train_dataset = []
			test_dataset = [] 
			training = True 
			total_dataset_size = 0
			# while True:

			# import tracemalloc
			# tracemalloc.start()

			if not param.il_load_loader_on:
				for k,file in enumerate(sorted(datadir)):
					
					print(file)

					if param.il_state_loss_on:
						dataset = load_orca_dataset_state_loss(
							file,
							param.r_comm,
							param.r_obs_sense,
							param.max_neighbors,
							param.max_obstacles,
							param.training_time_downsample)
					else:
						dataset = load_orca_dataset_action_loss(
							file,
							param.r_comm,
							param.r_obs_sense,
							param.max_neighbors,
							param.max_obstacles,
							param.training_time_downsample)
					
					if np.random.uniform(0, 1) <= param.il_test_train_ratio:
						train_dataset.extend(dataset)
					else:
						test_dataset.extend(dataset)

					print(len(train_dataset), len(test_dataset))

					if len(train_dataset) + len(test_dataset) > param.il_n_data:
						break

				print('Total Training Dataset Size: ',len(train_dataset))
				print('Total Testing Dataset Size: ',len(test_dataset))

				# # debug loading memory usage
				# snapshot = tracemalloc.take_snapshot()
				# top_stats = snapshot.statistics('lineno')

				# print("[ Top 10 ]")
				# for stat in top_stats[:10]:
				# 	print(stat)

				loader_train = make_loader(
					dataset=train_dataset,
					shuffle=True,
					batch_size=param.il_batch_size,
					n_data=param.il_n_data,
					name = "train")

				loader_test = make_loader(
					dataset=test_dataset,
					shuffle=True,
					batch_size=param.il_batch_size,
					n_data=param.il_n_data,
					name = "test")

			else:
				loader_train = load_loader("train",param.il_batch_size,True)
				loader_test  = load_loader("test",param.il_batch_size,True)

		# consensus dataset 
		elif "consensus" in param.il_load_dataset:

			observation_size = param.n_neighbors*param.agent_memory*param.state_dim_per_agent

			dataset = []
			for k,file in enumerate(glob.glob("../data/consensus/*.npy")):
				print(file)
				dataset.extend(load_consensus_dataset(file,param.n_neighbors,param.agent_memory))

			print('Total Dataset Size: ',len(dataset))
			loader_train,loader_test = make_orca_loaders(
				dataset=dataset,
				shuffle=False,
				batch_size=param.il_batch_size,
				test_train_ratio=param.il_test_train_ratio,
				n_data=param.il_n_data,
				max_neighbors=param.max_neighbors)

		# scp dataset 
		else:
			if param.il_state_loss_on:
				states = np.empty((0,env.n), dtype=np.float32)
				actions = np.empty((0,env.n), dtype=np.float32)
				for k,file in enumerate(glob.glob(param.il_load_dataset)):
					print(file)
					data = load_dataset(env, file)
					states = np.vstack([states, data[0:-2,0:env.n]])
					actions = np.vstack([actions, data[1:-1,0:env.n]])
			else:
				states = np.empty((0,env.n), dtype=np.float32)
				actions = np.empty((0,env.m), dtype=np.float32)
				for k,file in enumerate(glob.glob(param.il_load_dataset)):
					print(file)
					data = load_dataset(env, file)
					dist = np.linalg.norm(data[-1,0:env.n] - np.array([0,0,0,0]))
					if dist < 0.1:
						states = np.vstack([states, data[:,0:env.n]])
						actions = np.vstack([actions, data[:,env.n:env.n+env.m]])
					else:
						print("Skipping ", file)


			print('Total Dataset Size: ', states.shape[0])
			idx = int(param.il_test_train_ratio * states.shape[0])

			x_train = torch.from_numpy(states[0:idx])
			y_train = torch.from_numpy(actions[0:idx])
			x_test = torch.from_numpy(states[idx:-1])
			y_test = torch.from_numpy(actions[idx:-1])

			dataset_train = Data.TensorDataset(x_train, y_train)
			loader_train = Data.DataLoader(
				dataset=dataset_train,
				batch_size=param.il_batch_size,
				shuffle=True)
			dataset_test = Data.TensorDataset(x_test, y_test)
			loader_test = Data.DataLoader(
				dataset=dataset_test,
				batch_size=param.il_batch_size,
				shuffle=True)

	# make dataset from rl model 
	else:
		print('Making Dataset')
		x_train,y_train = make_dataset(param, env)
		x_test,y_test = make_dataset(param, env)
		dataset_train = Data.TensorDataset(x_train, y_train)
		loader_train = Data.DataLoader(
			dataset=dataset_train,
			batch_size=param.il_batch_size,
			shuffle=True)
		dataset_test = Data.TensorDataset(x_test, y_test)
		loader_test = Data.DataLoader(
			dataset=dataset_test, 
			batch_size=param.il_batch_size, 
			shuffle=True)


	# training 
	best_test_loss = Inf
	optimizer = torch.optim.Adam(model.parameters(), lr=param.il_lr, weight_decay = param.il_wd)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	for epoch in range(1,param.il_n_epoch+1):
		
		train_epoch_loss = train(param,env,model,optimizer,loader_train)
		test_epoch_loss = test(param,env,model,loader_test)
		scheduler.step(test_epoch_loss)

		if epoch%param.il_log_interval==0:
			print('epoch: ', epoch)
			print('   Train Epoch Loss: ', train_epoch_loss)
			print('   Test Epoch Loss: ', test_epoch_loss)
			if test_epoch_loss < best_test_loss:
				best_test_loss = test_epoch_loss
				print('      saving @ best test loss:', best_test_loss)
				torch.save(model,param.il_train_model_fn)

	# # debug loading memory usage
	# snapshot = tracemalloc.take_snapshot()
	# top_stats = snapshot.statistics('lineno')

	# print("[ Top 10 ]")
	# for stat in top_stats[:10]:
	# 	print(stat)



def make_dataset(param, env):
	model = torch.load(param.il_imitate_model_fn)
	times = param.sim_times
	states = []
	actions = []
	while len(states) < param.il_n_data:
		states.append(env.reset())
		for step, time in enumerate(times[:-1]):

			observations = env.observe()
			action = model.policy(observation)
			s_prime, _, done, _ = env.step(action)

			states.append(s_prime)
			if param.il_state_loss_on:
				actions.append(s_prime)
			else:
				actions.append(action.reshape(-1))
			
			if done:
				break

		if param.il_state_loss_on:
			actions.append(zeros(env.n))
		else:
			actions.append(zeros(env.m))

	states = states[0:param.il_n_data]
	actions = actions[0:param.il_n_data]
	return torch.tensor(states).float(),torch.tensor(actions).float()


def load_dataset(env, filename):
	data = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	return data[0:-2] # do not include last row (invalid action)

	# return 	torch.tensor(data[0:-2,1:1+env.n]).float(),
	# 		torch.tensor(data[0:-2,1+env.n:1+env.n+env.m]).float()






# def make_dataset(env):
# 	# model = PlainPID([2, 40], [4, 20])
# 	model = torch.load(param.il_imitate_model_fn)
# 	states = []
# 	actions = []
# 	for _ in range(param.il_n_data):
# 		state = array((
# 			env.env_state_bounds[0]*uniform(-1.,1.),
# 			env.env_state_bounds[1]*uniform(-1.,1.),
# 			env.env_state_bounds[2]*uniform(-1.,1.),
# 			env.env_state_bounds[3]*uniform(-1.,1.),         
# 			))
# 		action = model.policy(state)
# 		action = action.reshape((-1))
# 		states.append(state)
# 		actions.append(action)
# 	return torch.tensor(states).float(), torch.tensor(actions).float()

# def load_orca_dataset_state_loss(filename,neighborDist):
# 	data = np.load(filename)
# 	num_agents = int((data.shape[1] - 1) / 4)
# 	# loop over each agent and each timestep to find
# 	#  * current state
# 	#  * set of neighboring agents (storing their relative states)
# 	#  * label (i.e., desired control)
# 	dataset = []
# 	for t in range(data.shape[0]-1):
# 		for i in range(num_agents):
# 			state_i = data[t,i*4+1:i*4+5]
# 			sg_i = data[-1,i*4+1:i*4+5]
# 			neighbors = []
# 			for j in range(num_agents):
# 				if i != j:
# 					state_j = data[t,j*4+1:j*4+5]
# 					dist = np.linalg.norm(state_i[0:2] - state_j[0:2])
# 					if dist <= neighborDist:
# 						neighbors.append(state_i - state_j)
# 			# desired control is the velocity in the next timestep
# 			state_i_tp1 = data[t+1,i*4+1:i*4+5]
# 			dataset.append([state_i, sg_i, neighbors, state_i_tp1])
# 	print('Dataset Size: ',len(dataset))
# 	return dataset

# def load_orca_dataset_action_loss_old(filename,neighborDist):
# 	data = np.load(filename)
# 	num_agents = int((data.shape[1] - 1) / 4)
# 	# loop over each agent and each timestep to find
# 	#  * current state
# 	#  * set of neighboring agents (storing their relative states)
# 	#  * label (i.e., desired control)
# 	dataset = []

# 	# this new 
# 	Observation_Action_Pair = namedtuple('Observation_Action_Pair', ['observation', 'action']) 
# 	Observation = namedtuple('Observation',['relative_goal','relative_neighbors']) 

# 	for t in range(data.shape[0]-1):
# 		for i in range(num_agents):
# 			s_i = data[t,i*4+1:i*4+5]   # state i 
# 			s_g = data[-1,i*4+1:i*4+5]  # goal state i 
# 			relative_goal = s_g - s_i # relative goal 
# 			relative_neighbors = []
# 			for j in range(num_agents):
# 				if i != j:
# 					s_j = data[t,j*4+1:j*4+5] # state j
# 					dist = np.linalg.norm(state_i[0:2] - state_j[0:2])
# 					if dist <= neighborDist:
# 						relative_neighbors.append(s_j - s_i)
# 			# desired control is the velocity in the next timestep
# 			a = data[t+1, i*4+3:i*4+5]
# 			# this new
# 			o = Observation._make((relative_goal,relative_neighbors))			
# 			oa_pair = Observation_Action_Pair._make((o,a))
# 			dataset.append(oa_pair)
# 			# this old
# 			# dataset.append([sg_i-state_i, neighbors, u]) 
# 	print('Dataset Size: ',len(dataset))
# 	return dataset

# def make_orca_loaders(dataset=None,
# 	shuffle=True,
# 	batch_size=200,
# 	test_train_ratio=0.8,
# 	n_data=None):

# 	def make_loader(dataset):
# 		batch_x = []
# 		batch_y = []
# 		loader = [] 
# 		for step,data in enumerate(dataset):

# 			batch_x.append(data.observation)
# 			batch_y.append(data.action)

# 			if (step+1)%batch_size == 0 and step is not 0:
# 				loader.append([batch_x,batch_y])
# 				batch_x = []
# 				batch_y = []
# 		return loader

# 	if dataset is None:
# 		raise Exception('dataset not specified')
	
# 	loader_train = make_loader(train_dataset)
# 	loader_test = make_loader(test_dataset)
# 	return loader_train,loader_test