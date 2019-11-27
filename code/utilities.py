

import numpy as np 


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
	K = np.zeros((len(observations),n_agents,n_agents))

	for t,observation_t in enumerate(observations):
		k_t = controller.get_belief_topology(observation_t)

		for i_agent in range(n_agents):
			n_neighbors = len(k_t[i_agent])
			for j_agent in range(n_neighbors):
				K[t,i_agent,j_agent] = k_t[i_agent][j_agent]
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


