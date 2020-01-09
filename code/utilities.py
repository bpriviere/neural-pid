

import numpy as np 
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



def min_dist_circle_rectangle(circle_pos, circle_r, rect_tl, rect_br):
	# Find the closest point to the circle within the rectangle
	closest = np.clip(circle_pos, rect_tl, rect_br)
	# Calculate the distance between the circle's center and this closest point
	dist = np.linalg.norm(circle_pos - closest, axis=1)
	
	# return closest point in rectangle, and distance to circle
	return closest, dist