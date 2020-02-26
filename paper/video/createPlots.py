import glob
import os
import numpy as np
import yaml
import torch
import argparse
from scipy import spatial

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

def plot_state_space(map_data, data, t):
	# print("state space" + r["solver"])
	fig, ax = plt.subplots()
	# ax.set_title("State Space " + r["solver"])
	ax.set_aspect('equal')

	for o in map_data["map"]["obstacles"]:
		ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for x in range(-1,map_data["map"]["dimensions"][0]+1):
	# 	ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# 	ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for y in range(map_data["map"]["dimensions"][0]):
	# 	ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# 	ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

	num_agents = len(map_data["agents"])
	colors = []
	for i in range(num_agents):
		line = ax.plot(data[:,1+i*4], data[:,1+i*4+1],linestyle='dashed')
		color = line[0].get_color()
		# start = np.array(map_data["agents"][i]["start"]) + np.array([0.5,0.5])
		start = data[t,1+i*4:1+i*4+2]
		goal = np.array(map_data["agents"][i]["goal"])
		ax.add_patch(Circle(start, 0.2, alpha=0.5, color=color))
		ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))
		colors.append(color)

	return colors

def plot_observation(observation, color=None, has_action = True):
	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.set_xlim(-3,3)
	ax.set_ylim(-3,3)
	ax.set_autoscalex_on(False)
	ax.set_autoscaley_on(False)

	# print(observation)
	num_neighbors = int(observation[0])
	if has_action:
		num_obstacles = int((observation.shape[0]-5 - 2*num_neighbors)/2)
	else:
		num_obstacles = int((observation.shape[0]-3 - 2*num_neighbors)/2)

	# print(observation, num_neighbors, num_obstacles)

	robot_pos = np.array([0,0])

	ax.add_patch(Circle(robot_pos, 3.0, facecolor='gray', edgecolor='black', alpha=0.1))

	ax.add_patch(Circle(robot_pos, 0.2, facecolor=color, alpha=0.5))
	
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
	ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color=color))

	# plot action
	if has_action:
		plt.arrow(0,0,observation[-2],observation[-1])

	return fig

def get_observations(map_data, data, sampling = 100, Robs = 3, max_neighbors = 5):
	data = torch.from_numpy(data)

	obstacles = []
	for o in map_data["map"]["obstacles"]:
		obstacles.append(np.array(o) + np.array([0.5,0.5]))

	# for x in range(-1,map_data["map"]["dimensions"][0]+1):
	# 	obstacles.append(np.array([x,-1]) + np.array([0.5,0.5]))
	# 	obstacles.append(np.array([x,map_data["map"]["dimensions"][1]]) + np.array([0.5,0.5]))
	# for y in range(map_data["map"]["dimensions"][0]):
	# 	obstacles.append(np.array([-1,y]) + np.array([0.5,0.5]))
	# 	obstacles.append(np.array([map_data["map"]["dimensions"][0],y]) + np.array([0.5,0.5]))

	obstacles = np.array(obstacles)
	kd_tree_obstacles = spatial.KDTree(obstacles)

	num_agents = int((data.shape[1] - 1) / 4)
	dataset = []
	reached_goal = set()
	for t in range(0,data.shape[0]-1):
		if t%sampling != 0:
			continue

		# build kd-tree
		positions = np.array([data[t,i*4+1:i*4+3].numpy() for i in range(num_agents)])
		kd_tree_neighbors = spatial.KDTree(positions)

		for i in range(num_agents):
			# # skip datapoints where agents are just sitting at goal
			# if i in reached_goal:
			# 	continue

			s_i = data[t,i*4+1:i*4+3]   # state i 
			# s_g = data[-1,i*4+1:i*4+5]  # goal state i 
			s_g = torch.Tensor(map_data["agents"][i]["goal"]) + torch.Tensor([0.5,0.5])
			# print(s_g, data[-1,i*4+1:i*4+5])
			relative_goal = s_g - s_i   # relative goal
			# if we reached the goal, do not include more datapoints from this trajectory
			# if np.allclose(relative_goal, np.zeros(2)):
				# reached_goal.add(i)
			if relative_goal.norm() < 0.5:
				reached_goal.add(i)
			time_to_goal = data[-1,0] - data[t,0]

			# query visible neighbors
			_, neighbor_idx = kd_tree_neighbors.query(
				s_i[0:2].numpy(),
				k=max_neighbors+1,
				distance_upper_bound=Robs)
			if type(neighbor_idx) is not np.ndarray:
				neighbor_idx = [neighbor_idx]
			relative_neighbors = []
			for k in neighbor_idx[1:]: # skip first entry (self)
				if k < positions.shape[0]:
					relative_neighbors.append(data[t,k*4+1:k*4+3] - s_i)
				else:
					break

			# query visible obstacles
			_, obst_idx = kd_tree_obstacles.query(
				s_i[0:2].numpy(),
				k=max_neighbors,
				distance_upper_bound=Robs)
			if type(obst_idx) is not np.ndarray:
				obst_idx = [obst_idx]
			relative_obstacles = []
			for k in obst_idx:
				if k < obstacles.shape[0]:
					relative_obstacles.append(obstacles[k,:] - s_i[0:2].numpy())
				else:
					break

			num_neighbors = len(relative_neighbors)
			num_obstacles = len(relative_obstacles)

			obs_array = np.empty(3+2*num_neighbors+2*num_obstacles+2, dtype=np.float32)
			obs_array[0] = num_neighbors
			idx = 1

			# conditional normalization of relative goal
			dist = np.linalg.norm(relative_goal[0:2])
			if dist > Robs:
				relative_goal[0:2] = relative_goal[0:2] / dist * Robs

			obs_array[idx:idx+2] = relative_goal
			idx += 2
			# obs_array[4] = data.observation.time_to_goal
			for k in range(num_neighbors):
				obs_array[idx:idx+2] = relative_neighbors[k]
				idx += 2
			for k in range(num_obstacles):
				obs_array[idx:idx+2] = relative_obstacles[k]
				idx += 2
			# action: velocity
			obs_array[idx:idx+2] = data[t+1, i*4+3:i*4+5]
			idx += 2

			dataset.append((i,t,obs_array))

	return dataset


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("instance", help="input file containing instance (yaml)")
	parser.add_argument("schedule", help="input file containing trajectories (npy)")
	args = parser.parse_args()
	sampling = 50

	with open(args.instance) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	data = np.load(args.schedule)
	for t in range(0,data.shape[0]-1):
		if t%sampling != 0:
			continue
		colors = plot_state_space(map_data, data, t)
		plt.axis('off')
		plt.savefig("obs_central_f{}.png".format(int(t/sampling)), dpi=300, bbox_inches='tight')

	dataset = get_observations(map_data, data, sampling=sampling)
	for agent, t, obs in dataset:
		fig = plot_observation(obs, color=colors[agent])
		plt.axis('off')
		plt.savefig("obs_agent{}_f{}.png".format(agent, int(t/sampling)), dpi=300, bbox_inches='tight')


