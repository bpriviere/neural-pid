import glob
import os
import stats
import numpy as np
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


def add_scatter(pp, results, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	# find set of solvers
	solvers = set()
	for _, results in result_by_instance.items():
		for r in results:
			solvers.add(r["solver"])

	width = 0.8 / len(solvers)

	for k, solver in enumerate(sorted(solvers)):
		idx = 0
		x = []
		y = []
		for instance in sorted(result_by_instance):
			results = result_by_instance[instance]
			for r in results:
				if r["solver"] == solver:
					x.append(idx)
					y.append(r[key])
			idx += 1
		# ax.scatter(x,y,label=solver)
		ax.bar(np.array(x)+k*width, y, width, label=solver)

	ax.set_xticks(np.arange(len(result_by_instance)))
	# ax.set_xticklabels([instance for instance, _ in result_by_instance.items()])

	plt.legend()

	pp.savefig(fig)
	plt.close(fig)


def add_bar_agg(pp, results, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	# find set of solvers
	solvers = set()
	for _, results in result_by_instance.items():
		for r in results:
			solvers.add(r["solver"])

	# x = []
	# y = []
	for k, solver in enumerate(sorted(solvers)):

		agg = 0
		for _, results in result_by_instance.items():
			for r in results:
				if r["solver"] == solver:
					agg += r[key]
		ax.bar(k, agg)
		# x.append(k)
		# y.append(agg)
	# print(y)
	# ax.bar(x, y)

	ax.set_xticks(np.arange(len(solvers)))
	ax.set_xticklabels([solver for solver in sorted(solvers)])

	pp.savefig(fig)
	plt.close(fig)


def add_bar_chart(pp, results, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	y_pos = np.arange(len(results))
	ax.bar(y_pos, [d[key] for d in results])
	ax.set_xticks(y_pos)
	ax.set_xticklabels([r["solver"] for r in results])
	# ax.set_ylabel(key)

	pp.savefig(fig)
	plt.close(fig)


if __name__ == '__main__':

	result_by_instance = dict()

	for file in glob.glob("**/*obst6_agents4_ex000*.npy", recursive=True):
	# for file in glob.glob("**/*.npy", recursive=True):
		solver = os.path.dirname(file)
		# if solver not in ["EN", "ENwAPF", "orca"]:
		# 	continue
		instance = os.path.splitext(os.path.basename(file))[0]
		map_filename = "instances/{}.yaml".format(instance)
		result = stats.stats(map_filename, file)
		result["solver"] = solver
		if instance in result_by_instance:
			result_by_instance[instance].append(result)
		else:
			result_by_instance[instance] = [result]
		# print(file, solver, instance)

	pp = PdfPages("results.pdf")

	add_bar_agg(pp, result_by_instance, "num_agents_success", "# robots success")
	add_scatter(pp, result_by_instance, "percent_agents_reached_goal", "% reached goal")
	add_scatter(pp, result_by_instance, "num_collisions", "# collisions")

	for instance in sorted(result_by_instance):
		print(instance)
		results = result_by_instance[instance]

		add_bar_chart(pp, results, "percent_agents_reached_goal", instance + " (% reached goal)")
		add_bar_chart(pp, results, "num_collisions", instance + " (# collisions)")

		map_filename = "instances/{}.yaml".format(instance)
		with open(map_filename) as map_file:
			map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

		for r in results:
			print("state space" + r["solver"])
			fig, ax = plt.subplots()
			ax.set_title("State Space " + r["solver"])
			ax.set_aspect('equal')

			for o in map_data["map"]["obstacles"]:
				ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
			for x in range(-1,map_data["map"]["dimensions"][0]+1):
				ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
				ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
			for y in range(map_data["map"]["dimensions"][0]):
				ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
				ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

			data = np.load("{}/{}.npy".format(r["solver"], instance))
			num_agents = len(map_data["agents"])
			for i in range(num_agents):
				line = ax.plot(data[:,1+i*4], data[:,1+i*4+1])
				color = line[0].get_color()
				start = np.array(map_data["agents"][i]["start"])
				goal = np.array(map_data["agents"][i]["goal"])
				ax.add_patch(Circle(start + np.array([0.5,0.5]), 0.2, alpha=0.5, color=color))
				ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))

			pp.savefig(fig)
			plt.close(fig)



	pp.close()
