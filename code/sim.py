
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

def sim(param, env, controllers, visualize):

	def run_sim(controller, initial_state):
		states = np.zeros((len(times), env.n))
		actions = np.zeros((len(times) - 1,env.m))
		states[0] = env.reset(initial_state)
		reward = 0 
		for step, time in enumerate(times[:-1]):
			state = states[step]
			if param.pomdp_on:
				observation = env.observe()
				action = controller.policy(observation)
			else:
				action = controller.policy(state) 
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

	# initial conditions
	if False:
		# consensus
		# s0 = np.array([5,-1,0,2,0,1,1,0,0,-1,1,0,-2,0,-1.5])
		# orca 10 ring
		# s0 = np.array([50,0,0,0,40.4509,29.3893,0,0,15.4509,47.5528,0,0,-15.4509,47.5528,0,0,-40.4509,29.3893,0,0,-50,6.12323e-15,0,0,-40.4509,-29.3893,0,0,-15.4509,-47.5528,0,0,15.4509,-47.5528,0,0,40.4509,-29.3893,0,0])
		# orca 2 line 
		s0 = np.array([-20,0,0,0,20,0,0,0])
		initial_state = env.reset(s0)
	else:
		initial_state = env.reset()

	# run sim
	SimResult = namedtuple('SimResult', ['states', 'actions', 'steps', 'name'])
	sim_results = []
	for name, controller in controllers.items():
		result = SimResult._make(run_sim(controller, initial_state) + (name, ))
		sim_results.append(result)

	if param.sim_render_on:
		save_rl_env.render()
		save_il_env.render()
	
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
		for i_config in range(env.config_dim):
			fig,ax = plotter.make_fig()
			for agent in env.agents:
				ax.set_title(env.states_name[i_config])
				for result in sim_results:
					ax.plot(
						times[0:result.steps],
						result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)+i_config],
						label=result.name)

	# plot state space
	if param.multi_agent_sim:
		fig,ax = plotter.make_fig()
		for agent in env.agents:
			ax.set_title('State Space')
			for result in sim_results:
				ax.plot(
					result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)],
					result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)+1],
					label=result.name)

	# # extract gains
	# if param.controller_class in ['PID_wRef','PID'] and not isinstance(pid_controller,ZeroPolicy):
	# 	kp,kd = util.extract_gains(pid_controller,states_pid)
	# 	fig,ax = plotter.plot(times[1:step_pid+1],kp[0:step_pid,0],title='Kp pos')
	# 	fig,ax = plotter.plot(times[1:step_pid+1],kp[0:step_pid,1],title='Kp theta')
	# 	fig,ax = plotter.plot(times[1:step_pid+1],kd[0:step_pid,0],title='Kd pos')
	# 	fig,ax = plotter.plot(times[1:step_pid+1],kd[0:step_pid,1],title='Kd theta')

	# # extract reference trajectory
	# if param.controller_class in ['PID_wRef','Ref'] and not isinstance(pid_controller,ZeroPolicy):
	# 	ref_state = util.extract_ref_state(pid_controller, states_pid)
	# 	for i in range(env.n):
	# 		fig,ax = plotter.plot(times[1:step_pid+1],ref_state[0:step_pid,i],title="ref " + env.states_name[i])

	# visualize
	if visualize:
		# plotter.visualize(param, env, states_deeprl)
		env.visualize(sim_results[0].states[0:result.steps],0.01)

	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)
