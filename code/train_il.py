
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np 
import random 

from numpy import array, zeros, Inf
from numpy.random import uniform,seed
from torch.distributions import Categorical

from learning.pid_net import PID_Net
from learning.pid_wref_net import PID_wRef_Net
from learning.ref_net import Ref_Net
from learning.plain_pid import PlainPID
from learning.deepset import DeepSet 

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


def load_orca_dataset(filename,neighborDist):
	data = np.load(filename)
	num_agents = int((data.shape[1] - 1) / 4)
	# loop over each agent and each timestep to find
	#  * current state
	#  * set of neighboring agents (storing their relative states)
	#  * label (i.e., desired control)
	dataset = []
	for t in range(data.shape[0]-1):
		for i in range(num_agents):
			state_i = data[t,i*4+1:i*4+5]
			sg_i = data[-1,i*4+1:i*4+5]
			neighbors = []
			for j in range(num_agents):
				if i != j:
					state_j = data[t,j*4+1:j*4+5]
					dist = np.linalg.norm(state_i[0:2] - state_j[0:2])
					if dist <= neighborDist:
						neighbors.append(state_i - state_j)
			# desired control is the velocity in the next timestep
			u = data[t+1, i*4+3:i*4+5]
			dataset.append([state_i, sg_i, neighbors, u])
	print('Dataset Size: ',len(dataset))
	return dataset


def make_orca_loaders(dataset=None,
	shuffle=True,
	batch_size=200,
	test_train_ratio=0.8,
	n_data=None):

	def make_loader(dataset):
		batch_x = []
		batch_y = []
		loader = [] 
		for step,data in enumerate(dataset):
			for k,elem in enumerate(data):
				if k == 0:
					x = data[k]
				elif k == 1:
					x = np.concatenate((x,data[k]))
				elif k == 2:
					for neighbor in data[k]:
						x = np.concatenate((x,neighbor))
				elif k == 3:
					y = data[k]

			batch_x.append(x)
			batch_y.append(y)

			if step%batch_size == 0 and step is not 0:
				loader.append([batch_x[0:batch_size],batch_y[0:batch_size]])
				batch_x = []
				batch_y = []
		return loader

	if dataset is None:
		raise Exception('dataset not specified')
	
	if shuffle:
		random.shuffle(dataset)

	if n_data is not None:
		dataset = dataset[0:n_data]

	cutoff = int(test_train_ratio*len(dataset))
	train_dataset = dataset[0:cutoff]
	test_dataset = dataset[cutoff:]

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
			action = model.policy(states[-1])
			s_prime, _, done, _ = env.step(action)
			states.append(s_prime)
			actions.append(action.reshape(-1))
			if done:
				break
		actions.append(zeros(env.m))

	states = states[0:param.il_n_data]
	actions = actions[0:param.il_n_data]

	return torch.tensor(states).float(),torch.tensor(actions).float()


def train(param, model, loader):

	optimizer = torch.optim.Adam(model.parameters(), lr=param.il_lr)
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step

		# convert b_y if necessary
		if not isinstance(b_y, torch.Tensor):
			b_y = torch.from_numpy(np.array(b_y)).float()

		prediction = model(b_x)     # input x and predict based on x
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # apply gradients
		epoch_loss += loss 
	return epoch_loss/step


def test(model, loader):
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step

		# convert b_y if necessary
		if not isinstance(b_y, torch.Tensor):
			b_y = torch.from_numpy(np.array(b_y)).float()

		prediction = model(b_x)     # input x and predict based on x
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		epoch_loss += loss 
	return epoch_loss/step


def train_il(param, env):

	seed(1) # numpy random gen seed 
	torch.manual_seed(1)    # pytorch 

	# init model
	if param.controller_class is 'PID':
		model = PID_Net(env.n)
	elif param.controller_class is 'PID_wRef':
		model = PID_wRef_Net(env.n)
	elif param.controller_class is 'Ref':
		model = Ref_Net(env.n, env.m, param.kp, param.kd)
	elif param.controller_class is 'DeepSet':
		model = DeepSet(
			param.network_architecture_phi,
			param.network_architecture_rho,
			param.network_activation)
	else:
		print('Error in Train Gains, programmatic controller not recognized')
		exit()

	print("Case: ",param.env_case)
	print("Controller: ",param.controller_class)

	# datasets
	if param.il_load_dataset_on:
		dataset = load_orca_dataset("../baseline/orca/build/orca.npy",param.r_comm)
		loader_train,loader_test = make_orca_loaders(
			dataset=dataset,
			shuffle=True,
			batch_size=param.il_batch_size,
			test_train_ratio=param.il_test_train_ratio,
			n_data=None)
	else:
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

	best_test_loss = Inf
	for epoch in range(1,param.il_n_epoch+1):
		train_epoch_loss = train(param, model, loader_train)
		test_epoch_loss = test(model, loader_test)
		if epoch%param.il_log_interval==0:
			print('epoch: ', epoch)
			print('   Train Epoch Loss: ', train_epoch_loss)
			print('   Test Epoch Loss: ', test_epoch_loss)
			if test_epoch_loss < best_test_loss:
				best_test_loss = test_epoch_loss
				print('      saving @ best test loss:', best_test_loss)
				torch.save(model,param.il_train_model_fn)
