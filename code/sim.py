
# standard package
import torch
import gym 
import numpy as np 
import os

# my package
import plotter 
import utilities as util
from other_policy import ZeroPolicy

def sim(param, env, visualize):

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
			reward += env.reward()
			states[step + 1], _, done, _ = env.step(action)
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

	# get controllers
	if False:
		deeprl_controller = torch.load(param.sim_rl_model_fn)
		pid_controller = torch.load(param.sim_il_model_fn)
	else:
		# set to empty controllers
		deeprl_controller = torch.load(param.sim_il_model_fn)
		# pid_controller = ZeroPolicy(env)

	# initial conditions
	if True:
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
	states_deeprl, actions_deeprl, step_deeprl = run_sim(deeprl_controller, initial_state)
	save_rl_env = env
	initial_state = s0
	# states_pid, actions_pid, step_pid = run_sim(pid_controller, initial_state)
	# save_il_env = env

	if os.path.isfile(param.scp_fn):
		data_csv = np.loadtxt(param.scp_fn, delimiter=',')
		states_csv = data_csv[:,0:env.n]
		actions_csv = data_csv[:,env.n:env.n+env.m]

	if param.sim_render_on:
		save_rl_env.render()
		save_il_env.render()
	
	# plot time varying states
	if param.single_agent_sim:
		for i in range(env.n):
			fig, ax = plotter.plot(times[0:step_deeprl],states_deeprl[0:step_deeprl,i]) #,title=env.states_name[i])
			plotter.plot(times[0:step_pid],states_pid[0:step_pid,i], fig = fig, ax = ax)
			if os.path.isfile(param.scp_fn):
				plotter.plot(times[0:len(states_csv)],states_csv[:,i], fig = fig, ax = ax)
		for i in range(env.m):
			fig, ax = plotter.plot(times[1:step_deeprl+1],actions_deeprl[0:step_deeprl,i]) #,title=env.actions_name[i])
			plotter.plot(times[1:step_pid+1],actions_pid[0:step_pid,i], fig = fig, ax = ax)
			if os.path.isfile(param.scp_fn):
				plotter.plot(times[1:len(actions_csv)+1],actions_csv[:,i], fig = fig, ax = ax)
	
	elif param.multi_agent_sim:
		for i_config in range(env.config_dim):
			fig,ax = plotter.make_fig()
			for agent in env.agents:
				plotter.plot(
					times[0:step_deeprl],
					states_deeprl[0:step_deeprl,env.agent_idx_to_state_idx(agent.i)+i_config],
					fig=fig,ax=ax,title=env.states_name[i_config])

	# plot state space
	if param.multi_agent_sim:
		fig,ax = plotter.make_fig()
		for agent in env.agents:
			plotter.plot(
				states_deeprl[0:step_deeprl,env.agent_idx_to_state_idx(agent.i)],
				states_deeprl[0:step_deeprl,env.agent_idx_to_state_idx(agent.i)+1],
				fig=fig,ax=ax,title='State Space')	

	# extract gains
	if param.controller_class in ['PID_wRef','PID'] and not isinstance(pid_controller,ZeroPolicy):
		kp,kd = util.extract_gains(pid_controller,states_pid)
		fig,ax = plotter.plot(times[1:step_pid+1],kp[0:step_pid,0],title='Kp pos')
		fig,ax = plotter.plot(times[1:step_pid+1],kp[0:step_pid,1],title='Kp theta')
		fig,ax = plotter.plot(times[1:step_pid+1],kd[0:step_pid,0],title='Kd pos')
		fig,ax = plotter.plot(times[1:step_pid+1],kd[0:step_pid,1],title='Kd theta')

	# extract reference trajectory
	if param.controller_class in ['PID_wRef','Ref'] and not isinstance(pid_controller,ZeroPolicy):
		ref_state = util.extract_ref_state(pid_controller, states_pid)
		for i in range(env.n):
			fig,ax = plotter.plot(times[1:step_pid+1],ref_state[0:step_pid,i],title="ref " + env.states_name[i])

	# visualize
	if visualize:
		# plotter.visualize(param, env, states_deeprl)
		env.visualize(states_deeprl,0.1)

	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)
