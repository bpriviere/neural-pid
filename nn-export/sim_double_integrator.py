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
max_v = 0.5 # 0.5


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


	# -----------------------------------DI----------------------------
	# head-on
	start = np.array([
		[2.,0.0,0,0],
		[-2.0,0.0,0,0],
		])
	goal = np.array([
		[-2.0,0.0,0,0],
		[2.0,0.0,0,0],
		])
	obstacles = np.array([
		# [0,0.]
	])

	# raytheon exp1
	# start = np.array([
	# 	[0.8,0.9,0,0],
	# 	[-0.9,-0.5,0,0],
	# 	])
	# goal = np.array([
	# 	[-0.9,-0.1,0,0],
	# 	[1.2,0.8,0,0],
	# 	])
	# obstacles = np.array([
	# 	# [0.52,-0.53],
	# 	# [-0.1,1.12]
	# ])

	# # exp 3: ring
	# num_agents = 3
	# r = 1.0
	# theta = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
	# start = np.zeros((num_agents,4))
	# start[:,0] = r * np.cos(theta)
	# start[:,1] = r * np.sin(theta)
	# goal = -start
	# obstacles = np.array([
	# 	[0,0.0]
	# 	])	


	num_agents = len(start)

	dt = 0.025
	vel_dt = 2.0
	ts = np.arange(0,50,dt)
	result = np.zeros((len(ts), 6 * num_agents))
	dbg_result = np.zeros((len(ts), 3 * num_agents))
	for i in range(num_agents):
		idx = i*6
		result[0,idx:idx+4] = start[i]

	# simulation
	for k, t in enumerate(ts[0:-1]):
		print(t)
		for i in range(num_agents):
			idx = i*6
			p_i = result[k,idx:idx+2]
			v_i = result[k,idx+2:idx+4]
			s_i = result[k,idx:idx+4]
			# apply policy
			nnexport.nn_reset()
			
			for j in range(num_agents):
				if j != i:
					idxj = j*6
					p_j=result[k,idxj:idxj+2]
					relative_neighbor = p_j - p_i
					dist = np.linalg.norm(relative_neighbor)
					if dist < r_comm:
						s_j = result[k,idxj:idxj+4]
						# print(s_j - s_i)
						nnexport.nn_add_neighbor(s_j - s_i)

			for o in obstacles:
				relative_obstacle = o - p_i
				dist = np.linalg.norm(relative_obstacle)
				if dist < r_comm:
					nnexport.nn_add_obstacle(np.concatenate((relative_obstacle, -v_i)))

			relative_goal = goal[i] - s_i
			normg = np.linalg.norm(relative_goal[0:2])
			if normg > 0:
				relative_goal[0:2] = relative_goal[0:2] * np.min((r_comm/normg,1))
			print('relative_goal',relative_goal)
			print('r_comm',r_comm)

			# acceleration = nnexport.nn_eval(relative_goal)
			dbg = nnexport.nn_eval(relative_goal)
			acceleration = np.array(dbg)[0:2]
			dbg_result[k+1,i*3:(i+1)*3] = np.array(dbg)[2:5]
			# print(i, acceleration)

			result[k+1,idx:idx+2] = result[k,idx:idx+2] + result[k,idx+2:idx+4] * dt

			# result[k+1,idx+2:idx+4] = np.clip(result[k,idx+2:idx+4] + np.array(acceleration) * dt, -max_v, max_v)

			v = result[k,idx+2:idx+4] + np.array(acceleration) * dt
			normv = np.linalg.norm(v)
			if normv > 0:
				v = v * np.min((max_v / normv, 1))
			result[k+1,idx+2:idx+4] = v 

			result[k+1,idx+4:idx+6] = np.array(acceleration)

	# collision checker
	for k, t in enumerate(ts):
		for i in range(num_agents):
			idx_i = i*6
			p_i = result[k,idx_i:idx_i+2]
			for j in range(i+1, num_agents):
				idx_j = j*6
				p_j = result[k, idx_j:idx_j+2]
				dist = np.linalg.norm(p_j - p_i)
				if dist < 0.3:
					print("WARNING: collision between {} and {} at {} with dist {}".format(i,j,t,dist))


	fig, ax = plt.subplots()
	ax.set_title('State Space')
	ax.set_aspect('equal')
	# ax.set_xlim([-1,1.5])
	# ax.set_ylim([-1,1])

	# ax.set_xlim([-1.25,1.25])
	# ax.set_ylim([-0.7,1.5])

	for o in obstacles:
		ax.add_patch(Rectangle(o-0.5, 1.0, 1.0, facecolor='gray', alpha=0.5))

	for i in range(num_agents):
		idx = i*6
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


	# plot velocities
	fig, ax = plt.subplots()
	ax.set_title('velocities')

	for i in range(num_agents):
		idx = i*6
		vel = np.linalg.norm(result[:,idx+2:idx+4],axis=1)
		line = ax.plot(ts, vel)

	plt.show()

	# plot accelerations
	fig, ax = plt.subplots()
	ax.set_title('accelerations')

	for i in range(num_agents):
		idx = i*6
		acc = np.linalg.norm(result[:,idx+4:idx+6],axis=1)
		line = ax.plot(ts, acc)

	plt.show()



	fig, ax = plt.subplots()
	ax.set_title('alpha')

	for i in range(num_agents):
		idx = i*3
		line = ax.plot(ts, dbg_result[:,idx+0])

	plt.show()

	fig, ax = plt.subplots()
	ax.set_title('||pi||')

	for i in range(num_agents):
		idx = i*3
		line = ax.plot(ts, dbg_result[:,idx+1])

	plt.show()

	fig, ax = plt.subplots()
	ax.set_title('||b||')

	for i in range(num_agents):
		idx = i*3
		line = ax.plot(ts, dbg_result[:,idx+2])

	plt.show()