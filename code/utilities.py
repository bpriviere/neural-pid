

import numpy as np 
from scipy.linalg import block_diag
import torch 

def to_cvec(x):
	return np.reshape(x,(len(x),-1))

def extract_gains(controller, states):
	kp = np.zeros((len(states)-1,2))
	kd = np.zeros((len(states)-1,2))
	i = 0
	for state in states[1:]:
		kp[i] = controller.get_kp(state)
		kd[i] = controller.get_kd(state)
		i += 1
	return kp,kd

def extract_belief_topology(controller,observations):

	n_agents = len(observations[0])
	K = np.empty((len(observations),n_agents,n_agents))


	for t,observation_t in enumerate(observations):
		k_t = controller.get_belief_topology(observation_t)

		for i_agent in range(n_agents):
			# n_neighbors = len(k_t[i_agent])

			count = 0
			for j_agent in range(n_agents):

				if not i_agent == j_agent:
					K[t,i_agent,j_agent] = k_t[i_agent][count]
					count += 1 

	return K


def extract_ref_state(controller, states):
	ref_state = np.zeros((len(states)-1,4))
	for i, state in enumerate(states[1:]):
		ref_state[i] = controller.get_ref_state(state)
	return ref_state

def debug(variable):
	print(variable + '=' + repr(eval(variable)))

def debug_lst(variable_lst):
	for variable in variable_lst:
		debug(variable)


def torch_tile(a, dim, n_tile):
	# from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def rot_mat_2d(th):
	return np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])



def preprocess_transformation(dataset_batches):
	# input: 
	# 	- list of tuple of (observation, actions) pairs, numpy/pytorch supported
	# output: 
	# 	- list of tuple of (observation, actions) pairs, numpy arrays
	# 	- list of transformations 

	# TEMP 
	obstacleDist = 3.0 
	transformed_dataset_batches = []
	transformations_batches = []	
	for (dataset, classification) in dataset_batches:

		# dataset = [#n, sg-si, {sj-si}, {so-si}]

		if isinstance(dataset,torch.Tensor):
			dataset = dataset.detach().numpy()
		if isinstance(classification,torch.Tensor):
			classification = classification.detach().numpy()
				
		if dataset.ndim == 1:
			dataset = np.reshape(dataset,(-1,len(dataset)))
		if classification.ndim == 1:
			classification = np.reshape(classification,(-1,len(classification)))

		num_neighbors = int(dataset[0,0]) #int((x.size()[1]-4)/4)
		num_obstacles = int((dataset.shape[1]-5-4*num_neighbors)/2)

		idx_goal = np.arange(1,3,dtype=int)

		transformed_dataset = np.empty(dataset.shape)
		transformed_classification = np.empty(classification.shape)
		transformations = np.empty((dataset.shape[0],2,2))

		for k,row in enumerate(dataset):

			transformed_row = np.empty(row.shape)
			transformed_row[0] = row[0]

			# get goal 
			# s_gi = sg - si 
			s_gi = row[idx_goal]

			# get transformation 
			th = np.arctan2(s_gi[1],s_gi[0])
			
			R = rot_mat_2d(th)
			# R = rot_mat_2d(0)
			bigR = block_diag(R,R)

			# conditional normalization of relative goal
			dist = np.linalg.norm(s_gi[0:2])
			if dist > obstacleDist:
				s_gi[0:2] = s_gi[0:2] / dist * obstacleDist

			# transform goal 
			transformed_row[idx_goal] = np.matmul(bigR,s_gi)

			# get neighbors
			# transform neighbors 
			for j in range(num_neighbors):
				idx = 1+4+j*4+np.arange(0,4,dtype=int)
				s_ji = row[idx] 
				bigR = block_diag(R,R)
				transformed_row[idx] = np.matmul(bigR,s_ji)

			# get obstacles
			# transform neighbors 
			for j in range(num_obstacles):
				idx = 1+4+num_neighbors*4+j*2+np.arange(0,2,dtype=int)
				s_oi = row[idx] 
				transformed_row[idx] = np.matmul(R,s_oi)
			
			# transform action
			if classification is not None: 
				transformed_classification[k,:] = np.matmul(R,classification[k])
			transformed_dataset[k,:] = transformed_row
			transformations[k,:,:] = R

		transformed_dataset_batches.append((transformed_dataset,transformed_classification))
		transformations_batches.append(transformations)

	return transformed_dataset_batches, transformations_batches
