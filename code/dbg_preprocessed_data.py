import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

# load training data:
datadir = glob.glob("../preprocessed_data/batch_test*.npy")
num_examples = dict()
for file in datadir: 
	batch = np.load(file)
	num_neighbors = int(batch[0,0])
	num_obstacles = int((batch.shape[1] - 1 - 2 - num_neighbors*2 - 2) / 2)
	num_examples[(num_neighbors,num_obstacles)] =  batch.shape[0]

max_neighbors = np.max([nn for nn, _ in num_examples.keys()])
max_obstacles = np.max([no for _, no in num_examples.keys()])
num_examples_array = np.zeros((max_neighbors+1, max_obstacles+1))
for (nn, no), value in num_examples.items():
	num_examples_array[nn,no] = value

plt.imshow(num_examples_array / num_examples_array.sum(),cmap=matplotlib.cm.hot)
plt.colorbar()
plt.ylabel("# neighbors")
plt.xlabel("# obstacles")
plt.show()

print("# data: ", num_examples_array.sum())

exit()




pp = PdfPages("preprocessed_data_stats.pdf")

datadir = glob.glob("../preprocessed_data/batch_train*.npy")
velocity = []
closest_neighbor_dist = []
for file in datadir: 
	batch = np.load(file)

	relative_goal = batch[:,1:3]
	goal_dist = np.linalg.norm(relative_goal, axis=1)
	# plt.hist(goal_dist)

	num_neighbors = batch[0,0]
	if num_neighbors > 0:
		closest_neighbor = batch[:,3:5]
		dist = np.linalg.norm(closest_neighbor, axis=1)
		closest_neighbor_dist.extend(dist.tolist())

	# fig, ax = plt.subplots()
	# ax.set_title(file)
	# ax.hist(goal_dist)
	# pp.savefig(fig)
	# plt.close(fig)

	action = batch[:,-2:]
	vel = np.linalg.norm(action, axis=1)
	velocity.extend(vel.tolist())
	# fig, ax = plt.subplots()
	# ax.set_title("velocity")
	# ax.hist(vel)
	# pp.savefig(fig)
	# plt.close(fig)


print("# data ", len(velocity))
fig, ax = plt.subplots()
ax.set_title("velocity")
ax.hist(velocity)
pp.savefig(fig)
plt.close(fig)

print("# data with at least one neighbor: ", len(closest_neighbor_dist))
fig, ax = plt.subplots()
ax.set_title("dist to closest neighbor")
ax.hist(closest_neighbor_dist)
pp.savefig(fig)
plt.close(fig)


pp.close()

	# plt.show()
	# break



