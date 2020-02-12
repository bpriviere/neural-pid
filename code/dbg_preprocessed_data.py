import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


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



