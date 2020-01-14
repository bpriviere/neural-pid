import glob
import numpy as np
import hnswlib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

state_dim_per_agent = 2

def plot_obs(pp, observation, title=None):
	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.set_xlim(-3,3)
	ax.set_ylim(-3,3)
	ax.set_autoscalex_on(False)
	ax.set_autoscaley_on(False)
	ax.set_title(title)

	num_neighbors = int(observation[0])
	num_obstacles = int((observation.shape[0]-5 - 2*num_neighbors)/2)

	print(observation, num_neighbors, num_obstacles)

	robot_pos = np.array([0,0])
	ax.add_patch(Circle(robot_pos, 0.2, facecolor='b', alpha=0.5))
	
	idx = 3
	for i in range(num_neighbors):
		pos = observation[idx : idx+2] + robot_pos
		ax.add_patch(Circle(pos, 0.2, facecolor='gray', edgecolor='red', alpha=0.5))
		idx += 2

	for i in range(num_obstacles):
		pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
		ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
		# pos = observation[idx : idx+2] + robot_pos
		# ax.add_patch(Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
		idx += 2

	# plot goal
	goal = observation[1:3] + robot_pos
	ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color='blue'))

	# plot action
	plt.arrow(0,0,observation[-2],observation[-1])

	ax.add_patch(Circle(robot_pos, 3.0, facecolor='gray', edgecolor='black', alpha=0.1))

	# plt.show()
	pp.savefig(fig)
	plt.close(fig)

pp = PdfPages("index_test.pdf")

datadir = glob.glob("../preprocessed_data/batch_train*.npy")
for file in datadir:
	data = np.load(file)
	index_fn = "{}.index".format(file)
	num_neighbors = int(data[0,0])
	num_obstacles = int((data.shape[1] - 1 - state_dim_per_agent - num_neighbors*state_dim_per_agent - 2) / 2)
	print(file, num_neighbors, num_obstacles)

	# ignore first column (num_neighbors) and last two columns (actions)
	dim = data.shape[1] - 3 
	p = hnswlib.Index(space='l2', dim=dim)
	if os.path.exists(index_fn):
		p.load_index(index_fn)
	else:
		# see https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md for params
		p.init_index(max_elements = data.shape[0],ef_construction=100, M=16)
		p.add_items(data[:,1:1+dim])
		p.save_index("{}.index".format(file))

	print("Index ready")
	obs = data[10,1:1+dim]
	labels, distances = p.knn_query(obs, k=5)
	print(labels, distances)

	data2 = np.lib.format.open_memmap(file, mode='r')
	print(data2.dtype)
	for k, l in enumerate(labels[0]):
		print(l,data[l], data2[l])
		plot_obs(pp, data[l], str(k))

pp.close()

	# break



