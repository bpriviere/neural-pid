import glob
import os
import numpy as np
import yaml
import torch
from scipy import spatial
import subprocess

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

def plot_state_space(map_data, data):
	# print("state space" + r["solver"])
	fig, ax = plt.subplots()
	# ax.set_title("State Space " + r["solver"])
	ax.set_aspect('equal')

	for o in map_data["map"]["obstacles"]:
		ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
	for x in range(-1,map_data["map"]["dimensions"][0]+1):
		ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
		ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
	for y in range(map_data["map"]["dimensions"][0]):
		ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
		ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

	num_agents = len(map_data["agents"])
	colors = []
	for i in range(num_agents):
		line = ax.plot(data[:,1+i*4], data[:,1+i*4+1])
		color = line[0].get_color()
		start = np.array(map_data["agents"][i]["start"])
		goal = np.array(map_data["agents"][i]["goal"])
		ax.add_patch(Circle(start + np.array([0.5,0.5]), 0.2, alpha=0.5, color=color))
		ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))
		colors.append(color)

	return colors



if __name__ == '__main__':

	solvers = ['central', 'orcaR3','apf','exp1Empty_0', 'exp1Barrier_0']
	instance = "map_8by8_obst6_agents8_ex0000"

	map_filename = "../../results/singleintegrator/instances/{}.yaml".format(instance)
	with open(map_filename) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)


	import sys
	sys.path.insert(1, os.path.join(os.getcwd(),'../../results/singleintegrator'))
	import stats

	for solver in solvers:
		schedule_filename = "../../results/singleintegrator/{}/{}.npy".format(solver, instance)
		results = stats.stats(map_filename, schedule_filename)
		data = np.load(schedule_filename)
		colors = plot_state_space(map_data, data)
		plt.axis('off')
		output_fn = "{}.pdf".format(solver)
		plt.savefig(output_fn, bbox_inches='tight')
		# subprocess.run(["pdfcrop", output_fn, output_fn])
		print(solver, results["percent_agents_success"])


