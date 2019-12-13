import glob
import os
import stats
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

def add_bar_chart(pp, solvers, results, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	y_pos = np.arange(len(solvers))
	ax.bar(y_pos, [d[key] for d in results])
	ax.set_xticks(y_pos)
	ax.set_xticklabels(solvers)
	# ax.set_ylabel(key)

	pp.savefig(fig)
	plt.close(fig)


if __name__ == '__main__':

	result_by_instance = dict()
	for file in glob.glob("**/*.npy", recursive=True):
		solver = os.path.dirname(file)
		instance = os.path.splitext(os.path.basename(file))[0]
		if instance in result_by_instance:
			result_by_instance[instance].append((solver, file))
		else:
			result_by_instance[instance] = [(solver, file)]
		# print(file, solver, instance)

	pp = PdfPages("results.pdf")
	for instance, item in result_by_instance.items():
		map_filename = "../../baseline/centralized-planner/examples/{}.yaml".format(instance)

		results = []
		solvers = []
		for solver, file in item:
			results.append(stats.stats(map_filename, file))
			solvers.append(solver)

		add_bar_chart(pp, solvers, results, "num_agents_reached_goal", instance + " (reached goal)")
		add_bar_chart(pp, solvers, results, "num_agent_agent_collisions", instance + " (# collisions)")

	pp.close()



