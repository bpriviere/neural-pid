from run import run
from systems.cartpole import CartPole
from run_cartpole import CartpoleParam
from planning.rrt import rrt
from planning.scp import scp
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx

if __name__ == '__main__':
	param = CartpoleParam()
	param.env_case = 'Any90' #'SmallAngle','Swing90','Swing180', 'Any90'

	env = CartPole(param)
	G = nx.DiGraph()

	# 0, 0, 0, 0
	# 0.3, -0.75, 1, -4

	x0 = np.array([0, 0, 0, 0])
	states = []
	states.append(x0)
	idx = 0
	G.add_node(0)
	while True:
		if idx >= len(states) or idx >= 100:
			break
		x0 = states[idx]

		data = []
		for i in range(5000):
			env.reset(x0)
			for k in range(10):
				u = np.random.uniform(-param.rl_control_lim, param.rl_control_lim, 1)
				new_state, _, done, _ = env.step(u)
				if done:
					break

			if not done:
				data.append(new_state)

		data = np.array(data)

		if data.shape[0] > 2500:

			kmeans = KMeans(n_clusters=4).fit(data)
			print(kmeans.cluster_centers_)

			# check if we can reach any of the already existing cluster centers
			for idx2, state in enumerate(states):
				dist = np.min(np.linalg.norm(data - state, axis=1))
				if dist < 0.5:
					print("state {} reachable from {} (dist {})".format(state, x0, dist))
					G.add_edge(idx, idx2)

			for center in kmeans.cluster_centers_:
				states.append(center)
				G.add_node(len(states)-1)
				G.add_edge(idx, len(states)-1)
		else:
			print("state {} does not have enough neighbors".format(x0))

		# fig, ax = plt.subplots()
		# ax.plot(data[:,0], data[:,1], '.')
		# # ax.set_xticklabels(algorithms)
		# # ax.set_ylabel('Runtime [s]')
		# plt.show()

		idx += 1

	fig, ax = plt.subplots()
	nx.draw(G, arrows=True, options={'arrowstyle': '-|>'})
	plt.show()

	while True:
		changed = False
		for node in G.nodes():
			if G.out_degree(node) == 0:
				G.remove_node(node)
				changed = True
		if not changed:
			break

	fig, ax = plt.subplots()
	nx.draw(G, arrows=True, options={'arrowstyle': '-|>'})
	plt.show()



