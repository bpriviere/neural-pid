import numpy as np
# import scipy
# import scipy.spatial
# import nmslib
import hnswlib


import matplotlib.pyplot as plt

def rrt(param, env):

	batch_size = 250
	num_actions_per_step = 5
	check_goal_iter = 1000
	sample_goal_iter = 250
	eps = 0.5
	num_actions_per_steer = 1

	# initialize with start state
	x0 = env.reset()
	data = np.empty((50000, 4 + 1 + 1))
	data[0,0:4] = x0
	data[0,4] = -1
	data[0,5] = 0

	# goal state
	xf = param.ref_trajectory[:,-1]

	print("Plan from: {} to {}".format(x0, xf))

	# index = nmslib.init(method='hnsw', space='l2')
	# index.addDataPoint(0, data[0])
	# index.createIndex(print_progress=False)

	index = hnswlib.Index(space='l2', dim=data.shape[1]-2)
	# ef_construction - controls index search speed/build speed tradeoff
	#
	# M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
	# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
	index.init_index(max_elements=data.shape[0], ef_construction=100, M=16)

	# Controlling the recall by setting ef:
	# higher ef leads to better accuracy, but slower search
	index.set_ef(10)

	index.add_items(data[0:1,0:4])

	i = 1
	last_index_update_idx = 1
	no_solution_count = 0
	while i < data.shape[0] and no_solution_count < 500:
		# randomly sample a state
		if i % check_goal_iter == 0:
			x = xf
		else:
			x = np.random.uniform(-env.env_state_bounds, env.env_state_bounds)

		# find closest state in tree
		# idx = np.argmin(np.linalg.norm(data[0:i] - x, axis=1))
		# x_near = data[idx]
		# ids, distances = index.knnQuery(x, k=1)
		# x_near = data[ids[0]]

		ids, distances = index.knn_query(x, k=1)
		x_near_idx = int(ids[0][0])
		x_near = data[x_near_idx]
		# if idx != ids[0][0]:
		# 	idx2 = int(ids[0][0])
		# 	print(idx, ids, distances, np.linalg.norm(data[idx:idx+1] - x, axis=1), np.linalg.norm(data[idx2:idx2+1] - x, axis=1))

		# steer
		best_u = None
		best_dist = None
		best_state = None
		for l in range(num_actions_per_steer):
			# randomly generate a control signal
			u = np.random.uniform(-param.rl_control_lim, param.rl_control_lim, 1)

			# forward propagate
			env.reset(x_near[0:4])
			for k in range(num_actions_per_step):
				new_state, _, done, _ = env.step(u)
				if done:
					break

			if not done:
				dist = np.linalg.norm(new_state - x)
				if best_u is None or dist < best_dist:
					best_u = u
					best_state = new_state
					best_dist = dist
		# print(x, x_near, u, new_state)

		# check if state is valid
		if best_u is not None:
			no_solution_count = 0
			# tree.append(Motion(new_state, m_near))
			data[i,0:4] = best_state
			data[i,4] = x_near_idx
			data[i,5] = best_u
			# index.addDataPoint(i, data[i])
			# index.createIndex(print_progress=False)
			i += 1

			if i % batch_size == 0:
				print(i)
				index.add_items(data[last_index_update_idx:i, 0:4])
				last_index_update_idx = i
				# index.addDataPointBatch(data[i-batch_size:i], ids=range(i-batch_size,i+1))
				# index.createIndex(print_progress=False)

			if i % check_goal_iter == 0:
				ids, distances = index.knn_query(xf, k=1)
				dist = np.sqrt(distances[0][0])
				print("Distance to goal: ", dist)
				if dist <= eps:
					print("Found goal!")
					break
		else:
			no_solution_count += 1


	index.add_items(data[last_index_update_idx:i, 0:4])

	# find the best state with respect to the goal
	xf = param.ref_trajectory[:,-1]
	ids, distances = index.knn_query(xf, k=1)
	idx = int(ids[0][0])
	states = []
	actions = []
	while idx >= 0:
		x_near = data[idx]
		idx = int(x_near[4])
		states.append(x_near[0:4])
		actions.append(x_near[5])

	states.reverse()
	actions.reverse()
	del actions[0]
	actions.append(0)

	print(states)

	result = np.empty(((len(states) - 1) * num_actions_per_step + 1, 5))
	for i, (state, action) in enumerate(zip(states, actions)):
		result[i*num_actions_per_step,0:4] = state
		result[i*num_actions_per_step,4] = action
		if i < len(states) - 1:
			env.reset(state)
			for k in range(num_actions_per_step):
				state, _, _, _ = env.step(action)
				result[i*num_actions_per_step+k,0:4] = state
				result[i*num_actions_per_step+k, 4] = action

	# print(result)
	np.savetxt(param.rrt_fn, result, delimiter=',')

	# # compute reward
	# env.reset(result[0,0:4])
	# for row in result[0:-2]:
	# 	state, _, _, _ = env.step(row[4])
	# 	print(row[4], state)
	# print("Reward: ", env.reward(), " state: ", env.state)

	# # runtime plot
	# fig, ax = plt.subplots()
	# ax.plot(data[:,0], data[:,1], '*')
	# # ax.set_xticklabels(algorithms)
	# # ax.set_ylabel('Runtime [s]')
	# plt.show()

	# fig, ax = plt.subplots()
	# ax.plot(result[:,0])
	# ax.plot(result[:,1])
	# # ax.set_xticklabels(algorithms)
	# # ax.set_ylabel('Runtime [s]')
	# plt.show()

	return states, actions


