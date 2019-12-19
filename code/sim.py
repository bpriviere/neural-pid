
# standard package
import torch
import gym 
import numpy as np 
import os
import glob
from collections import namedtuple

# my package
import plotter
from matplotlib.patches import Rectangle
import utilities as util
from other_policy import ZeroPolicy, LCP_Policy
from learning.ppo_v2 import PPO


def run_sim(param, env, controller, initial_state):
	states = np.zeros((len(param.sim_times), env.n))
	actions = np.zeros((len(param.sim_times)-1,env.m))
	observations = [] 
	reward = 0 

	env.reset(initial_state)
	states[0] = np.copy(env.s)
	for step, time in enumerate(param.sim_times[:-1]):
		print('t: {}/{}'.format(time,param.sim_times[-1]))

		state = states[step]
		observation = env.observe()

		if param.env_name is 'Consensus' and (isinstance(controller, LCP_Policy) or isinstance(controller,PPO)):
			observation = env.unpack_observations(observation)

		action = controller.policy(observation,env.transformations)
		next_state, r, done, _ = env.step(action)
		reward += r
		
		states[step + 1] = next_state
		actions[step] = action.flatten()
		observations.append(observation)

		if done:
			break

	print('reward: ',reward)
	env.close()
	return states, observations, actions, step


def sim(param, env, controllers, initial_state, visualize):

	# environment
	times = param.sim_times
	device = "cpu"

	if initial_state is None:
		initial_state = env.reset()

	# run sim
	SimResult = namedtuple('SimResult', ['states', 'observations', 'actions', 'steps', 'name'])
	
	for name, controller in controllers.items():
		print("Running simulation with " + name)
		print("Initial State: ", initial_state)
		if hasattr(controller, 'policy'):
			result = SimResult._make(run_sim(param, env, controller, initial_state) + (name, ))
		else:
			observations = [] 
			result = SimResult._make((controller.states, observations, controller.actions, controller.steps, name))
		sim_results = []
		sim_results.append(result)

		# plot state space
		if param.env_name in ['SingleIntegrator','DoubleIntegrator'] :
			fig,ax = plotter.make_fig()
			ax.set_title('State Space')
			ax.set_aspect('equal')

			for o in env.obstacles:
				ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

			for agent in env.agents:
				
				line = ax.plot(result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)], 
					result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)+1])
				color = line[0].get_color()

				# num_obstacles = (result.observations.shape[1] - result.observations[0][0] - 5) / 2
				# print(num_obstacles)
# 
				# print(result.observations)

				plotter.plot_circle(result.states[1,env.agent_idx_to_state_idx(agent.i)],
					result.states[1,env.agent_idx_to_state_idx(agent.i)+1],param.r_agent,fig=fig,ax=ax,color=color)
				plotter.plot_square(result.states[-1,env.agent_idx_to_state_idx(agent.i)],
					result.states[-1,env.agent_idx_to_state_idx(agent.i)+1],param.r_agent,fig=fig,ax=ax,color=color)
				plotter.plot_square(agent.s_g[0],agent.s_g[1],param.r_agent,angle=45,fig=fig,ax=ax,color=color)

			# draw state for each time step
			robot = 0
			for step in np.arange(0, result.steps, 10):
				fig,ax = plotter.make_fig()
				ax.set_title('State at t={} for robot={}'.format(times[step], robot))
				ax.set_aspect('equal')

				# plot all obstacles
				for o in env.obstacles:
					ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

				# plot overall trajectory
				line = ax.plot(result.states[0:result.steps,env.agent_idx_to_state_idx(robot)], 
					result.states[0:result.steps,env.agent_idx_to_state_idx(robot)+1],"--")
				color = line[0].get_color()

				# plot current position
				plotter.plot_circle(result.states[step,env.agent_idx_to_state_idx(robot)],
					result.states[step,env.agent_idx_to_state_idx(robot)+1],param.r_agent,fig=fig,ax=ax,color=color)

				# plot current observation
				observation = result.observations[step][0][0]
				num_neighbors = int(observation[0])
				num_obstacles = int((observation.shape[0]-5 - 4*num_neighbors)/2)

				robot_pos = result.states[step,env.agent_idx_to_state_idx(robot):env.agent_idx_to_state_idx(robot)+2]
				for i in range(num_obstacles):
					pos = observation[5+i*2 : 5+i*2+2] + robot_pos - np.array([0.5,0.5])
					ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))

				# plot goal
				goal = observation[1:3] + robot_pos
				ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color=color))


		elif param.env_name == 'Consensus' and param.sim_render_on:
			env.render()
		elif param.sim_render_on:
			env.render()
	
		# plot time varying states
		if param.single_agent_sim:
			for i in range(env.n):
				fig, ax = plotter.subplots()
				ax.set_title(env.states_name[i])
				for result in sim_results:
					ax.plot(times[0:result.steps], result.states[0:result.steps,i],label=result.name)
				ax.legend()

			for i, name in enumerate(env.deduced_state_names):
				fig, ax = plotter.subplots()
				ax.set_title(name)
				for result in sim_results:
					deduce_states = np.empty((result.steps, len(env.deduced_state_names)))
					for j in range(result.steps):
						deduce_states[j] = env.deduce_state(result.states[j])

					ax.plot(times[0:result.steps], deduce_states[:,i],label=result.name)
				ax.legend()

			for i in range(env.m):
				fig, ax = plotter.subplots()
				ax.set_title(env.actions_name[i])
				for result in sim_results:
					ax.plot(times[0:result.steps], result.actions[0:result.steps,i],label=result.name)
				ax.legend()

		elif param.env_name == 'Consensus':
			for i_config in range(1): #range(env.state_dim_per_agent):
				fig,ax = plotter.make_fig()
				# ax.set_title(env.states_name[i_config])
				for agent in env.agents:
					if env.good_nodes[agent.i]:
						color = 'blue'
					else:
						color = 'red'
					for result in sim_results:
						ax.plot(
							times[0:result.steps],
							result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)+i_config],
							label=result.name,
							color=color)
				ax.axhline(
					env.desired_ave,
					label='desired',
					color='green')

		elif param.env_name in ['SingleIntegrator','DoubleIntegrator']:
			for i_config in range(env.state_dim_per_agent):
				fig,ax = plotter.make_fig()
				ax.set_title(env.states_name[i_config])
				for agent in env.agents:
					for result in sim_results:
						ax.plot(
							times[1:result.steps],
							result.states[1:result.steps,env.agent_idx_to_state_idx(agent.i)+i_config],
							label=result.name)


		# plot time varying actions
		if param.env_name in ['SingleIntegrator','DoubleIntegrator']:
			for i_config in range(env.action_dim_per_agent):
				fig,ax = plotter.make_fig()
				ax.set_title(env.actions_name[i_config])
				for agent in env.agents:
					for result in sim_results:
						ax.plot(
							times[1:result.steps],
							result.actions[1:result.steps,agent.i*env.action_dim_per_agent+i_config],
							label=result.name)
				
		# extract gains
		if name in ['PID_wRef','PID']:
			controller = controllers['IL']
			for result in sim_results:
				if result.name == 'IL':
					break
			kp,kd = util.extract_gains(controller,result.states[0:result.steps])
			fig,ax = plotter.plot(times[1:result.steps],kp[0:result.steps,0],title='Kp pos')
			fig,ax = plotter.plot(times[1:result.steps],kp[0:result.steps,1],title='Kp theta')
			fig,ax = plotter.plot(times[1:result.steps],kd[0:result.steps,0],title='Kd pos')
			fig,ax = plotter.plot(times[1:result.steps],kd[0:result.steps,1],title='Kd theta')

		# extract reference trajectory
		if name in ['PID_wRef','Ref']:
			controller = controllers['IL']
			for result in sim_results:
				if result.name == 'IL':
					break
			ref_state = util.extract_ref_state(controller, result.states)
			for i in range(env.n):
				fig,ax = plotter.plot(times[1:result.steps+1],ref_state[0:result.steps,i],title="ref " + env.states_name[i])

		# extract belief topology
		if name in ['IL'] and param.env_name is 'Consensus':
			for result in sim_results:
				if result.name == 'IL':
					break

			fig,ax = plotter.make_fig()
			belief_topology = util.extract_belief_topology(controller, result.observations)
			label_on = True
			if param.n_agents*param.n_agents > 10:
				label_on = False
			for i_agent in range(param.n_agents):
				for j_agent in range(param.n_agents):
					if not i_agent == j_agent and env.good_nodes[i_agent]:
						if label_on:
							plotter.plot(times[0:result.steps+1],belief_topology[:,i_agent,j_agent],
								fig=fig,ax=ax,label="K:{}{}".format(i_agent,j_agent))
						else:
							plotter.plot(times[0:result.steps+1],belief_topology[:,i_agent,j_agent],
								fig=fig,ax=ax)


	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)

	# visualize
	if visualize:
		# plotter.visualize(param, env, states_deeprl)
		env.visualize(sim_results[0].states[0:result.steps],0.1)
