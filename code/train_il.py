
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np 
import random 
import glob

from numpy import array, zeros, Inf
from numpy.random import uniform,seed
from torch.distributions import Categorical
from collections import namedtuple

from learning.pid_net import PID_Net
from learning.pid_wref_net import PID_wRef_Net
from learning.ref_net import Ref_Net
from learning.empty_net import Empty_Net
from learning.barrier_net import Barrier_Net
from learning.nl_el_net import NL_EL_Net
from learning.consensus_net import Consensus_Net

def load_orca_dataset_action_loss(filename,neighborDist):
	data = np.load(filename)
	num_agents = int((data.shape[1] - 1) / 4)
	dataset = []
	Observation_Action_Pair = namedtuple('Observation_Action_Pair', ['observation', 'action']) 
	Observation = namedtuple('Observation',['relative_goal','relative_neighbors']) 
	for t in range(data.shape[0]-1):
		for i in range(num_agents):
			s_i = data[t,i*4+1:i*4+5]   # state i 
			s_g = data[-1,i*4+1:i*4+5]  # goal state i 
			relative_goal = s_g - s_i   # relative goal 
			relative_neighbors = []
			for j in range(num_agents):
				if i != j:
					s_j = data[t,j*4+1:j*4+5] # state j
					dist = np.linalg.norm(s_i[0:2] - s_j[0:2])
					if dist <= neighborDist:
						relative_neighbors.append(s_j - s_i)
			o = Observation._make((relative_goal,relative_neighbors))
			a = data[t+1, i*4+3:i*4+5] # desired control is the velocity in the next timestep
			oa_pair = Observation_Action_Pair._make((o,a))
			dataset.append(oa_pair)
	print('Dataset Size: ',len(dataset))
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

def make_orca_loaders(dataset=None,n_data=None,test_train_ratio=None,shuffle=False,batch_size=None):

	def make_loader(dataset):
		batch_x = []
		batch_y = []
		loader = [] 
		for step,data in enumerate(dataset):

			batch_x.append(data.observation)
			batch_y.append(data.action)

			if (step+1)%batch_size == 0 and step is not 0:
				loader.append([batch_x,batch_y])
				batch_x = []
				batch_y = []
		return loader

	if dataset is None:
		raise Exception('dataset not specified')
	
	if shuffle:
		random.shuffle(dataset)

	if n_data is not None and n_data < len(dataset):
		dataset = dataset[0:n_data]

	if test_train_ratio is not None:
		cutoff = int(test_train_ratio*len(dataset))
		train_dataset = dataset[0:cutoff]
		test_dataset = dataset[cutoff:]
	else:
		raise Exception('test train ratio not specified')

	loader_train = make_loader(train_dataset)
	loader_test = make_loader(test_dataset)
	return loader_train,loader_test	

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


def train(param,env,model,loader):

	optimizer = torch.optim.Adam(model.parameters(), lr=param.il_lr)
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0

	for step, (b_x, b_y) in enumerate(loader): # for each training step

		# convert b_y if necessary
		if not isinstance(b_y, torch.Tensor):
			b_y = torch.from_numpy(np.array(b_y)).float()

		prediction = model(b_x)     # input x and predict based on x
		
		# print('prediction: ', prediction.shape)
		# print('b_x: ', len(b_x))
		# print('b_y: ', b_y.shape)
		# exit()

		# if param.il_state_loss_on:
		# 	prediction_a = prediction
		# 	prediction = torch.zeros((b_y.shape))
		# 	for k,a in enumerate(prediction_a): 
		# 		prediction[k,:] = env.next_state_training_state_loss(b_x[k],a)

		# print('preloss')
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		# print('postloss')

		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # apply gradients
		
		epoch_loss += float(loss)
	return epoch_loss/step


def test(param,env,model,loader):
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step

		# convert b_y if necessary
		if not isinstance(b_y, torch.Tensor):
			b_y = torch.from_numpy(np.array(b_y)).float()

		prediction = model(b_x)     # input batch state and predict batch action

		if param.il_state_loss_on:
			prediction_a = prediction
			prediction = torch.zeros((b_y.shape))
			for k,a in enumerate(prediction_a): 
				prediction[k,:] = env.next_state_training_state_loss(b_x[k],a)

		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		epoch_loss += float(loss)
	return epoch_loss/step


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
				datadir = glob.glob("../data/singleintegrator/centralplanner/*.npy")

			dataset = []
			for k,file in enumerate(datadir):
				print(file)
				if param.il_state_loss_on:
					dataset.extend(load_orca_dataset_state_loss(file,param.r_comm))
				else:
					dataset.extend(load_orca_dataset_action_loss(file,param.r_comm))
				print(len(dataset))

				if len(dataset) > param.il_n_data:
					break

			print('Total Dataset Size: ',len(dataset))
			loader_train,loader_test = make_orca_loaders(
				dataset=dataset,
				shuffle=True,
				batch_size=param.il_batch_size,
				test_train_ratio=param.il_test_train_ratio,
				n_data=param.il_n_data)

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
				n_data=param.il_n_data)

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
	for epoch in range(1,param.il_n_epoch+1):
		train_epoch_loss = train(param,env,model,loader_train)
		test_epoch_loss = test(param,env,model,loader_test)
		if epoch%param.il_log_interval==0:
			print('epoch: ', epoch)
			print('   Train Epoch Loss: ', train_epoch_loss)
			print('   Test Epoch Loss: ', test_epoch_loss)
			if test_epoch_loss < best_test_loss:
				best_test_loss = test_epoch_loss
				print('      saving @ best test loss:', best_test_loss)
				torch.save(model,param.il_train_model_fn)


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