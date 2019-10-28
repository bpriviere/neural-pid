

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

def extract_ref_state(controller, states):
	ref_state = np.zeros((len(states)-1,4))
	for i, state in enumerate(states[1:]):
		ref_state[i] = controller.get_ref_state(state)
	return ref_state


