
# my package
from param import Param
from run import run
from systems.consensus import Consensus
from run_consensus import ConsensusParam
from other_policy import LCP_Policy, WMSR_Policy
import plotter 

# standard package
import numpy as np 
from torch import nn as nn 
from torch import tanh
from torch.nn import LeakyReLU
import torch 
import os 


def run_sim(param, env, controller, initial_state):
	
	observation_size = (param.n_neighbors+1)*param.agent_memory*param.state_dim_per_agent
	times = param.sim_times
	states = np.zeros((len(times), env.n))
	actions = np.zeros(( (len(times)-1)*(param.n_agents-param.n_malicious), param.action_dim_per_agent))
	observations = np.zeros(( (len(times)-1)*(param.n_agents-param.n_malicious), observation_size))

	
	env.reset(initial_state)
	states[0] = initial_state.initial_values
	count = 0 
	for step, time in enumerate(times[:-1]):
		state = states[step]
		observation = env.observe()
		# observation = env.unpack_observation_temp_for_prev_rl_model(observation)
		observation = env.unpack_observations(observation)

		action = controller.policy(observation) 
		states[step + 1], r, done, _ = env.step(action)

		# print('step: {}'.format(step))
		# print('states[{}]: {}'.format(step, states[step]))

		for i in range(param.n_agents):
			if env.good_nodes[i]:
				# print('action[{}]: {}'.format(i, action[i]))
				# print('observation[{}]: {}'.format(i,observation[i]))
				observations[count] = observation[i]
				actions[count] = action[i]
				count += 1

		if done:
			break

	env.close()
	return observations, actions, states, step

if __name__ == '__main__':
	param = ConsensusParam()
	env = Consensus(param)

	# temp
	# param.agent_memory = 2 

	observation_size = param.n_neighbors*param.agent_memory*param.state_dim_per_agent
	action_size = param.action_dim_per_agent
	times = param.sim_times 

	data = np.empty((1,observation_size+action_size))
	controller = torch.load(param.il_imitate_model_fn)
	# controller = torch.load(param.sim_rl_model_fn)
	# controller = torch.load(param.rl_train_best_model_fn)
	
	first_pass = True
	trajectory_rollout_count = 0
	while first_pass or len(data) < param.il_n_data:
		trajectory_rollout_count += 1

		x0 = env.reset()
		while not env.bad_nodes[2]:
			x0 = env.reset()

		observations, actions, states, steps = run_sim(param, env, controller, initial_state = x0)
		oa_pair = np.hstack((observations, actions))
		
		if first_pass:
			data = oa_pair
			first_pass = False
		else:
			data = np.vstack((data, oa_pair))

		# # plotting
		if True:
			env.render()

			for i_config in range(1): #range(env.state_dim_per_agent):
				fig,ax = plotter.make_fig()
				# ax.set_title(env.states_name[i_config])			
				for agent in env.agents:
					if env.good_nodes[agent.i]:
						color = 'blue'
					else:
						color = 'red'
					ax.plot(
						times[0:steps],
						states[0:steps,env.agent_idx_to_state_idx(agent.i)+i_config],
						color=color)
				ax.axhline(
					env.desired_ave,
					label='desired',
					color='green')
		
		print('Number of Trajectories:', trajectory_rollout_count)
		print('Data Shape: ', data.shape)

	file = os.path.join(param.il_load_dataset,'rl_data.npy')
	np.save(file, data)

	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)
