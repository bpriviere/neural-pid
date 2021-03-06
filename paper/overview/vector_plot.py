
# standard 
import sys
import os
import glob
import numpy as np
import yaml
import argparse
import torch 
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages

# hack for package import 
sys.path.insert(1, os.path.join(os.getcwd(),'../code'))
sys.path.insert(1, os.path.join(os.getcwd(),'../code/examples'))
print(os.getcwd())

# my packages
from other_policy import APF, Empty_Net_wAPF
from run_singleintegrator import SingleIntegratorParam 
from systems.singleintegrator import SingleIntegrator

plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

# from https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
def is_collision_circle_rectangle(circle_pos, circle_r, rect_tl, rect_br):
	# Find the closest point to the circle within the rectangle
	closest = np.clip(circle_pos, rect_tl, rect_br)
	# Calculate the distance between the circle's center and this closest point
	dist = np.linalg.norm(circle_pos - closest)
	# If the distance is less than the circle's radius, an intersection occurs
	return dist + 1e-4 < circle_r


def plot_policy_vector_field(fig,ax,policy,map_data,i,param):
	
	obstacles = map_data["map"]["obstacles"]
	agents = [np.array(agent["start"]) + 0.5 for agent in map_data["agents"]]
	transformation = [[np.eye(2)]]
	dx = 0.5 #param.vector_plot_dx

	# X = np.arange(0,map_data["map"]["dimensions"][0]+dx,dx)
	# Y = np.arange(0,map_data["map"]["dimensions"][1]+dx,dx)
	# U = np.zeros((len(Y),len(X)))
	# V = np.zeros((len(Y),len(X)))
	# C = np.zeros((len(Y),len(X)))

	X = []
	Y = []
	U = []
	V = []
	C = []

	for i_x,x in enumerate(np.arange(0,map_data["map"]["dimensions"][0]+dx,dx)):
		for i_y,y in enumerate(np.arange(0,map_data["map"]["dimensions"][1]+dx,dx)):
			if not collision((x,y),obstacles,agents):

				o_xy = get_observation_i_at_xy_from_data(map_data,x,y,i,param)
				a = policy.policy(o_xy)
				print(len(X), len(Y))

				X.append(x)
				Y.append(y)
				U.append(a[0][0])
				V.append(a[0][1])
				C.append(np.linalg.norm( np.array([a[0][0],a[0][1]])))

	# normalize arrow length
	# U = U / np.sqrt(U**2 + V**2);
	# V = V / np.sqrt(U**2 + V**2);

	im = ax.quiver(X,Y,U,V,C,scale_units='xy')
	# fig.colorbar(im)


def collision(p,o_lst,agents):
	p = np.array(p)
	for o in o_lst:
		o = np.array(o)
		if is_collision_circle_rectangle(p, 0.15, o, o+1.0):
			return True
	for a in agents:
		if np.linalg.norm(p - a) < 0.35:
			return True
	return False


def get_observation_i_at_xy_from_data(map_data,x,y,i,param):

	# o = [#n, g^i-(x,y), {p^j - (x,y)}_neighbors, {p^j - (x,y)}_obstaclescenter ]
	# data = [ t, {(s,a)}_agents ] 

	p = np.array([x,y])
	nn = len(map_data["agents"])
	no = len(map_data["map"]["obstacles"])
	r_comm = param.r_comm 
	max_neighbors = param.max_neighbors
	max_obstacles = param.max_obstacles

	goal = np.array(map_data["agents"][i]["goal"])
	relative_goal = goal - p + 0.5
	scale = r_comm/np.linalg.norm(relative_goal)
	if scale < 1:
		relative_goal = scale*relative_goal

	relative_neighbors = []
	for j in range(nn):
		if not j == i:
			idx = 1 + 4*j + np.arange(0,2) 
			s_j = np.array(map_data["agents"][j]["start"]) + 0.5
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


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-barrier", action='store_true')
	parser.add_argument("-empty", action='store_true')
	parser.add_argument("-empty_wAPF", action='store_true')
	parser.add_argument("-current", action='store_true')
	parser.add_argument("-i", "--instance")
	parser.add_argument("-a","--agent")
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse_args()

	# input:
	#   - policy 
	#   - instance

	param = SingleIntegratorParam()
	env = SingleIntegrator(param)

	if args.barrier:
		policy_fn = '../results/singleintegrator/exp1Barrier_0/il_current.pt'
		policy = torch.load(policy_fn)
	elif args.empty:
		policy_fn = '../models/singleintegrator/empty.pt'
		policy = torch.load(policy_fn)
	elif args.current:
		policy_fn = '../models/singleintegrator/il_current.pt'
		policy = torch.load(policy_fn)
	elif args.empty_wAPF:
		policy_fn = '../models/singleintegrator/il_current.pt'
		policy = torch.load(policy_fn)
		policy = Empty_Net_wAPF(param,env,policy)
	else:
		exit('no policy recognized')

	if args.instance:
		instance_fn = args.instance
	else:
		# instance = "map_8by8_obst6_agents4_ex0000"
		instance = "map_8by8_obst6_agents4_ex0002"		
		# instance = "map_8by8_obst6_agents4_ex0010"
		print('default instance file: {}'.format(instance))
		instance_fn = "../results/singleintegrator/instances/{}.yaml".format(instance)

	with open(instance_fn) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)
	num_agents = len(map_data["agents"])


	if args.agent:
		agent = args.agent
	else:
		agent = 0 
		print('default agent index {}'.format(agent))


	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.set_xlim((0,map_data["map"]["dimensions"][0]))
	ax.set_ylim((0,map_data["map"]["dimensions"][1]))

	plot_policy_vector_field(fig,ax,policy,map_data,agent,param)

	for o in map_data["map"]["obstacles"]:
		ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for x in range(-1,map_data["map"]["dimensions"][0]+1):
	# 	ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# 	ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# for y in range(map_data["map"]["dimensions"][0]):
	# 	ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
	# 	ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

	for j in range(num_agents):
		# if not agent == j:
		agent_j_p = np.array(map_data["agents"][j]["start"]) + 0.5
		ax.add_patch(Circle(agent_j_p, 0.15, alpha=0.5, color='gray'))
		if agent == j:
			goal = np.array(map_data["agents"][j]["goal"])
			ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color='blue'))

	# plt.show(fig)

	plt.axis('off')
	plt.savefig("policy.svg", bbox_inches='tight')