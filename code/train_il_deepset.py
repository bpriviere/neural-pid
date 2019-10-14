
import torch
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np

neighborDist = 15


def load_dataset(filename):
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
			neighbors = []
			for j in range(num_agents):
				if i != j:
					state_j = data[t,j*4+1:j*4+5]
					dist = np.linalg.norm(state_i[0:2] - state_j[0:2])
					if dist <= neighborDist:
						neighbors.append(state_i - state_j)
			# desired control is the velocity in the next timestep
			u = data[t+1, i*4+3:i*4+5]
			dataset.append([state_i, neighbors, u])
	print(len(dataset))


if __name__ == '__main__':
	load_dataset("../baseline/orca/build/orca_ring10.npy")