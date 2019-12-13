import numpy as np
import argparse
import yaml

def stats(map_filename, schedule_filename):
	data = np.load(schedule_filename)

	with open(map_filename) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	# find goal times
	goal_times = []
	num_agents_reached_goal = 0
	for i, agent in enumerate(map_data["agents"]):
		goal = np.array([0.5,0.5]) + np.array(agent["goal"])
		distances = np.linalg.norm(data[:,(i*4+1):(i*4+3)] - goal, axis=1)
		lastIdx = np.max(np.argwhere(distances > 0.05))
		if lastIdx < data.shape[0] - 1:
			goal_time = data[lastIdx,0]
			num_agents_reached_goal += 1
		else:
			print("Warning: Agent {} did not reach its goal! Last Dist: {}".format(i, distances[-1]))
			goal_time = float('inf')
		goal_times.append(goal_time)
	goal_times = np.array(goal_times)

	# Sum of cost:
	soc = np.sum(goal_times)

	# makespan
	makespan = np.max(goal_times)

	# control effort (here: single integrator => velocity)
	control_effort = 0
	for i, agent in enumerate(map_data["agents"]):
		control_effort += np.sum(np.abs(data[:,i*4+3]))
		control_effort += np.sum(np.abs(data[:,i*4+4]))
	control_effort *= (data[1,0] - data[0,0])

	# Collisions
	num_agents = len(map_data["agents"])
	# min_dist = float('inf')
	num_agent_agent_collisions = 0
	for i in range(num_agents):
		for j in range(i+1, num_agents):
			pos_i = data[:,(i*4+1):(i*4+3)]
			pos_j = data[:,(j*4+1):(j*4+3)]
			distances = np.linalg.norm(pos_i - pos_j, axis=1)
			num_agent_agent_collisions += np.count_nonzero(distances < 0.4)
			# min_dist = min(min_dist, np.min(distances))

	result = dict()
	# result["min_dist"] = min_dist
	result["sum_time"] = soc
	result["makespan"] = makespan
	result["control_effort"] = control_effort
	result["num_agents_reached_goal"] = num_agents_reached_goal
	result["num_agent_agent_collisions"] = num_agent_agent_collisions

	return result


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("map", help="input file containing map")
	parser.add_argument("schedule")
	args = parser.parse_args()

	print(stats(args.map, args.schedule))
