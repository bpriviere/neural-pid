
# standard package
import torch
import gym 
import numpy as np 
import os
import glob
from collections import namedtuple

# my package
import plotter 
import utilities as util
from other_policy import ZeroPolicy

def sim(param, env, controllers, initial_state, visualize):

	def run_sim(controller, initial_state):
		states = np.zeros((len(times), env.n))
		actions = np.zeros((len(times)-1,env.m))
		states[0] = env.reset(initial_state)
		reward = 0 
		for step, time in enumerate(times[:-1]):
			state = states[step]
			if param.pomdp_on:
				observation = env.observe()
			else:
				observation = state
							
			action = controller.policy(observation) 
			states[step + 1], r, done, _ = env.step(action)
			reward += r
			actions[step] = action.flatten()
			if done:
				break

		print('reward: ',reward)
		env.close()
		return states, actions, step

	# -------------------------------------------

	# environment
	times = param.sim_times
	device = "cpu"

	if initial_state is None:
		initial_state = env.reset()

	print("Initial State: ", initial_state)

	# run sim
	SimResult = namedtuple('SimResult', ['states', 'actions', 'steps', 'name'])
	
	for name, controller in controllers.items():
		print("Running simulation with " + name)
		if hasattr(controller, 'policy'):
			result = SimResult._make(run_sim(controller, initial_state) + (name, ))
		else:
			result = SimResult._make((controller.states, controller.actions, controller.steps, name))
		sim_results = []		
		sim_results.append(result)

		if param.sim_render_on:
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

		elif param.multi_agent_sim:
			for i_config in range(1): #range(env.state_dim_per_agent):
				fig,ax = plotter.make_fig()
				ax.set_title(env.states_name[i_config])			
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
				
		# # plot state space
		# if param.multi_agent_sim:
		# 	fig,ax = plotter.make_fig()
		# 	for agent in env.agents:
		# 		ax.set_title('State Space')
		# 		for result in sim_results:
		# 			ax.plot(
		# 				result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)],
		# 				result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)+1],
		# 				label=result.name)

		# extract gains
		if param.il_controller_class in ['PID_wRef','PID']:
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
		if param.il_controller_class in ['PID_wRef','Ref']:
			controller = controllers['IL']
			for result in sim_results:
				if result.name == 'IL':
					break
			ref_state = util.extract_ref_state(controller, result.states)
			for i in range(env.n):
				fig,ax = plotter.plot(times[1:result.steps+1],ref_state[0:result.steps,i],title="ref " + env.states_name[i])

	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)

	# visualize
	if visualize:
		# plotter.visualize(param, env, states_deeprl)
		env.visualize(sim_results[0].states[0:result.steps],0.1)
