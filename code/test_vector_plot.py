

import glob
import os
import stats
import numpy as np
import yaml

import torch 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


# some parameters
r_comm = 3 
max_neighbors = 1
max_obstacles = 1 
dx = 0.25


def plot_policy_vector_field(fig,ax,policy,map_data,data,i):
	
	obstacles = map_data["map"]["obstacles"]
	transformation = [[np.eye(2)]]

	X = np.arange(0,map_data["map"]["dimensions"][0]+dx,dx)
	Y = np.arange(0,map_data["map"]["dimensions"][1]+dx,dx)
	U = np.zeros((len(X),len(Y)))
	V = np.zeros((len(X),len(Y)))
	C = np.zeros((len(X),len(Y)))

	o = [] 
	for i_x,x in enumerate(X):
		for i_y,y in enumerate(Y):
			if not collision((x,y),obstacles):

				o_xy = get_observation_i_at_xy_from_data(data,map_data,x,y,i)
				a = policy.policy(o_xy,transformation)

				U[i_y,i_x] = a[0][0]
				V[i_y,i_x] = a[0][1]
				C[i_y,i_x] = np.linalg.norm( np.array([a[0][0],a[0][1]]))

	ax.quiver(X,Y,U,V,C)


def collision(p,o_lst):
	return False


def get_observation_i_at_xy_from_data(data,map_data,x,y,i):

	# o = [#n, g^i-(x,y), {p^j - (x,y)}_neighbors, {p^j - (x,y)}_obstaclescenter ]
	# data = [ t, {(s,a)}_agents ] 

	p = np.array([x,y])
	nn = len(map_data["agents"])
	no = len(map_data["map"]["obstacles"])

	goal = np.array(map_data["agents"][i]["goal"])
	relative_goal = goal - p + 0.5
	scale = r_comm/np.linalg.norm(relative_goal)
	if scale < 1:
		relative_goal = scale*relative_goal

	relative_neighbors = []
	for j in range(nn):
		if not j == i:
			idx = 1 + 4*j + np.arange(0,2) 
			s_j = data[idx] 
			if np.linalg.norm(s_j-p) < r_comm:
				relative_neighbors.append(s_j - p)
	if len(relative_neighbors)>max_neighbors:
		relative_neighbors = sorted(relative_neighbors, key=lambda x: np.linalg.norm(x))
		relative_neighbors = relative_neighbors[0:max_neighbors]

	relative_obstacles = []
	for o in map_data["map"]["obstacles"]:
		p_o = np.array(o,dtype=float) + 0.5
		relative_obstacles.append(p_o - p) 
	if len(relative_obstacles)>max_neighbors:
		relative_obstacles = sorted(relative_obstacles, key=lambda x: np.linalg.norm(x))
		relative_obstacles = relative_obstacles[0:max_obstacles]

	o_array = np.empty((1+2+len(relative_neighbors)*2 + len(relative_obstacles)*2))
	o_array[0] = len(relative_neighbors)
	o_array[1:3] = relative_goal 
	idx = 3 
	for rn in relative_neighbors:
		o_array[idx:idx+2] = rn
		idx += 2
	for ro in relative_obstacles:
		o_array[idx:idx+2] = ro
		idx += 2
	observation = [np.reshape(o_array,(1,len(o_array)))]
	return observation



if __name__ == '__main__':

	# input:
	#   - policy 
	#   - instance  

	policy = 'barrier'
	instance = "map_8by8_obst6_agents4_ex0002"

	# 
	if not policy in ['empty','barrier','empty_wAPF']:
		exit('policy not recognized: {}'.format(policy))

	# load data
	data_fn = "../results/singleintegrator/{}/{}.npy".format(policy,instance)
	data = np.load(data_fn)
	
	# load map 
	instance_fn = "../results/singleintegrator/instances/{}.yaml".format(instance)
	with open(instance_fn) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)
	print(map_data)
	num_agents = len(map_data["agents"])

	# load policy
	policy_fn = '../models/singleintegrator/{}.pt'.format(policy)
	policy = torch.load(policy_fn)

	# which agent to show 
	i = 0 

	t_idx = np.arange(0,data.shape[0],100)
	# t_array = data[t_idx,0]
	t_array = [0]

	# fig,ax = plt.subplots()
	# ax.set_aspect('equal')
	# ax.set_xlim((-1,9))
	# ax.set_ylim((-1,9))
	# for o in map_data["map"]["obstacles"]:
	# 	ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for x in range(-1,map_data["map"]["dimensions"][0]+1):
	# 	ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# 	ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for y in range(map_data["map"]["dimensions"][0]):
	# 	ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# 	ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for j in range(num_agents):
	# 	if not i == j:
	# 		color = 'blue'
	# 	else:
	# 		color='black'
	# 	idx = 1 + 4*j + np.arange(0,2)
	# 	line = ax.plot(data[:,idx[0]],data[:,idx[1]],color=color)
	# 	start = np.array(map_data["agents"][j]["start"])
	# 	goal = np.array(map_data["agents"][j]["goal"])
	# 	ax.add_patch(Circle(start + np.array([0.5,0.5]), 0.2, alpha=0.5, color=color))
	# 	ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))		

	for t in t_array:

		fig, ax = plt.subplots()
		ax.set_aspect('equal')
		ax.set_xlim((-1,9))
		ax.set_ylim((-1,9))

		plot_policy_vector_field(fig,ax,policy,map_data,data[t,:],i)

		for o in map_data["map"]["obstacles"]:
			ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
		for x in range(-1,map_data["map"]["dimensions"][0]+1):
			ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
			ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
		for y in range(map_data["map"]["dimensions"][0]):
			ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
			ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

		for j in range(num_agents):
			if not i == j:
				color = 'blue'
			else:
				color = 'black'
				start = np.array(map_data["agents"][i]["start"])
				goal = np.array(map_data["agents"][i]["goal"])
				ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))

			idx = 1 + 4*j + np.arange(0,2)
			agent_j_p = data[t,idx]
			ax.add_patch(Circle(agent_j_p, 0.2, alpha=0.5, color=color))	

	plt.show(fig)