import glob
import os
import numpy as np
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle

import nnexport

plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

# some parameters
r_comm = 3
max_neighbors = 5
max_obstacles = 5 


def collision(p,o_lst):
	return False


def policy(map_data,x,y,i):

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
			s_j = map_data["agents"][j]["start"]
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

	# apply policy
	nnexport.nn_reset()
	for n in relative_neighbors:
		nnexport.nn_add_neighbor(n)
	for o in relative_obstacles:
		nnexport.nn_add_obstacle(o)
	return nnexport.nn_eval(relative_goal)

def randomGenPositions(num_agents):
	start = np.empty((num_agents,2))
	for i in range(num_agents):
		while True:
			pos = np.random.uniform(low=[-1.0,-0.5],high=[1.0,1.2],size=2)
			collision = False
			for j in range(0, num_agents):
				dist = np.linalg.norm(pos - start[j])
				if dist < 0.3:
					collision = True
			if not collision:
				start[i] = pos
				break
	return start


if __name__ == '__main__':

	# # exp1
	# start = np.array([
	# 	[0.92,1.0],
	# 	[-1.0,0.23],
	# 	# [1.4,-0.3],
	# 	])
	# goal = np.array([
	# 	[-1.0,0.23],
	# 	[0.92,1.0],
	# 	# [-1.4,0.2],
	# 	])
	# obstacles = np.array([
	# 	[-0.43079,1.26188],
	# 	[0.0172923,-0.366721],
	# ])

	# # exp 2
	# start = np.array([
	# 	[1.0,1.1],
	# 	[-0.8,-0.55],
	# 	# [1.4,-0.3],
	# 	])
	# goal = np.array([
	# 	[-0.7,-0.55],
	# 	[0.8,0.5],
	# 	])
	# obstacles = np.array([
	# 	[-0.06,0.4],
	# 	[0.52,-0.59],
	# 	[-0.7,-1.5],
	# ])

	# # exp 3: ring
	# num_agents = 7
	# r = 1.0
	# theta = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
	# start = np.empty((num_agents,2))
	# start[:,0] = r * np.cos(theta)
	# start[:,1] = r * np.sin(theta)
	# goal = -start
	# obstacles = np.array([[0,0.0]])

	# offset = np.array([0,0.3])
	# start += offset
	# goal += offset
	# obstacles += offset

	# print(goal)
	# exit()


	# # exp 4: random movement
	# num_agents = 8
	# start = randomGenPositions(num_agents)
	# goal = randomGenPositions(num_agents)

	# obstacles = np.array([
	# ])

	# start = np.array([
	# 	[1.0,0.4],
	# 	[-1.0,0.4],
	# 	# [1.4,-0.3],
	# 	])
	# goal = np.array([
	# 	[-1.0,0],
	# 	[1.0,-0.4],
	# 	# [-1.4,0.2],
	# 	])

	# obstacles = np.array([
	# 	[0.1,-0.8],
	# 	[-0.3,0.8],
	# ])

	# start = np.array([
	# 	[1.3,0.0],
	# 	[-1.2,-0.8],
	# 	# [1.4,-0.3],
	# 	])
	# goal = np.array([
	# 	[-1.6,0],
	# 	[1.3,-0.4],
	# 	# [-1.4,0.2],
	# 	])

	# obstacles = np.array([
	# 	[-0.2,-1.0],
	# 	[-0.8,0.0],
	# 	# [-0.2,-2.0],
	# ])



	# # Raytheon exp 1
	# start = np.array([
	# 	[0.8,0.9],
	# 	[-0.9,-0.5],
	# 	])
	# goal = np.array([
	# 	[-0.9,-0.1],
	# 	[1.2,0.8],
	# 	])
	# obstacles = np.array([
	# 	[0.52,-0.53],
	# 	[-0.06,0.43]
	# ])

	# # Raytheon exp 1
	# start = np.array([
	# 	[0.8,0.9],
	# 	[-0.9,-0.5],
	# 	])
	# goal = np.array([
	# 	[-0.9,-0.1],
	# 	[1.2,0.8],
	# 	])
	# obstacles = np.array([
	# 	[0.52,-0.53],
	# 	[-0.1,1.12]
	# ])

	# head-on
	start = np.array([
		[1.0,0.0],
		[-1.0,0.0],
		])
	goal = np.array([
		[-1.0,0.0],
		[1.0,0.0],
		])
	obstacles = np.array([
	])


	num_agents = len(start)

	dt = 0.05
	vel_dt = 2.0
	ts = np.arange(0,30,dt)
	result = np.zeros((len(ts), 4 * num_agents))
	for i in range(num_agents):
		idx = i*4
		result[0,idx:idx+2] = start[i]

	# simulation
	for k, t in enumerate(ts[0:-1]):
		print(t)
		for i in range(num_agents):
			idx = i*4
			p_i = result[k,idx:idx+2]
			# apply policy
			nnexport.nn_reset()
			
			for j in range(num_agents):
				if j != i:
					idxj = j*4
					p_j=result[k,idxj:idxj+2]
					relative_neighbor = p_j - p_i
					dist = np.linalg.norm(relative_neighbor)
					if dist < r_comm:
						nnexport.nn_add_neighbor(relative_neighbor)

			for o in obstacles:
				relative_obstacle = o - p_i
				dist = np.linalg.norm(relative_obstacle)
				if dist < r_comm:
					nnexport.nn_add_obstacle(relative_obstacle)

			relative_goal = goal[i] - p_i
			velocity = nnexport.nn_eval(relative_goal)
			print(i, velocity)
			result[k+1,idx:idx+2] = result[k,idx:idx+2] + np.array(velocity) * dt
			result[k+1,idx+2:idx+4] = np.array(velocity)

	# collision checker
	for k, t in enumerate(ts):
		for i in range(num_agents):
			idx_i = i*4
			p_i = result[k,idx_i:idx_i+2]
			for j in range(i+1, num_agents):
				idx_j = j*4
				p_j = result[k, idx_j:idx_j+2]
				dist = np.linalg.norm(p_j - p_i)
				if dist < 0.3:
					print("WARNING: collision between {} and {} at {} with dist {}".format(i,j,t,dist))


	fig, ax = plt.subplots()
	ax.set_title('State Space')
	ax.set_aspect('equal')
	# ax.set_xlim([-1,1.5])
	# ax.set_ylim([-1,1])

	ax.set_xlim([-1.25,1.25])
	ax.set_ylim([-0.7,1.5])

	for o in obstacles:
		ax.add_patch(Rectangle(o-0.5, 1.0, 1.0, facecolor='gray', alpha=0.5))

	for i in range(num_agents):
		idx = i*4
		line = ax.plot(result[:,idx+0], result[:,idx+1],alpha=0.5)

		color = line[0].get_color()

		# plot velocity vectors:
		X = []
		Y = []
		U = []
		V = []
		for k in np.arange(0,len(ts),int(vel_dt/dt)):
			p_i = result[k,idx:idx+2]
			v_i = result[k,idx+2:idx+4]
			X.append(p_i[0])
			Y.append(p_i[1])
			U.append(v_i[0])
			V.append(v_i[1])

		ax.quiver(X,Y,U,V,angles='xy', scale_units='xy',scale=0.5,color=color,width=0.005)

		# plot start location
		ax.add_artist(patches.Circle(result[0,idx:idx+2],radius=0.15,color=color))

		# plot goal location
		ax.add_artist(patches.Rectangle(goal[i]-0.15,height=0.3,width=0.3,color=color))

	plt.show()