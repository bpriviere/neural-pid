
# standard package
import torch
import gym 
import numpy as np 
import os

# my package
import plotter 

def sim(param, env, visualize):

	def run_sim(controller, initial_state):
		states = np.zeros((len(times), env.n))
		actions = np.zeros((len(times) - 1, env.m))
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
			actions[step] = np.squeeze(action)
			if done:
				break
			if param.sim_render_on: # and step%20==0:
				env.render()
		print('reward: ',reward)
		env.close()
		return states, actions, step

	def extract_gains(controller, states):
		kp = np.zeros((len(times)-1,2))
		kd = np.zeros((len(times)-1,2))
		i = 0
		for state in states[1:]:
			kp[i] = controller.get_kp(state)
			kd[i] = controller.get_kd(state)
			i += 1
		return kp,kd

	def extract_ref_state(controller, states):
		ref_state = np.zeros((len(times)-1,4))
		for i, state in enumerate(states[1:]):
			ref_state[i] = controller.get_ref_state(state)
		return ref_state

	class ZeroPolicy:
		def policy(self,state):
			return np.zeros((env.m))

	class BoringConsensusPolicy:
		def policy(self,observation):
			a = np.zeros((env.m))
			dt = env.times[env.time_step] - env.times[env.time_step-1]			
			for agent in env.agents:
				# observation_i = {s^j - s^i} \forall j in N^i
				# a[agent.i] = sum(observation[agent.i])*dt
				a[agent.i] = sum(observation[agent.i])*0.1				
			return a

	# -------------------------------------------

	# environment
	times = param.sim_times
	device = "cpu"

	if False:
		# get controllers
		deeprl_controller = torch.load(param.sim_rl_model_fn)
		pid_controller = torch.load(param.sim_il_model_fn)
	else:
		# set to empty controllers
		deeprl_controller = BoringConsensusPolicy()
		pid_controller = ZeroPolicy()

	if False:
		s0 = np.array([-2.5,1,0,0])
		initial_state = env.reset(s0)
	else:
		initial_state = env.reset()

	# run sim
	states_deeprl, actions_deeprl, step_deeprl = run_sim(deeprl_controller, initial_state)
	save_rl_env = env
	states_pid, actions_pid, step_pid = run_sim(pid_controller, initial_state)
	save_il_env = env

	if os.path.isfile(param.scp_fn):
		data_csv = np.loadtxt(param.scp_fn, delimiter=',')
		states_csv = data_csv[:,0:env.n]
		actions_csv = data_csv[:,env.n:env.n+env.m]

	if param.sim_render_on:
		plotter.plot_ss(save_rl_env,states_deeprl)
		plotter.plot_ss(save_il_env,states_pid)
	
	# time varying states
	if False:
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
	else:
		fig,ax = plotter.make_fig()
		for agent in env.agents:
			plotter.plot(times[0:step_deeprl],states_deeprl[0:step_deeprl,agent.i],fig=fig,ax=ax)

	# extract gains
	if param.controller_class in ['PID_wRef','PID'] and not isinstance(pid_controller,ZeroPolicy):
		kp,kd = extract_gains(pid_controller,states_pid)
		fig,ax = plotter.plot(times[1:step_pid+1],kp[0:step_pid,0],title='Kp pos')
		fig,ax = plotter.plot(times[1:step_pid+1],kp[0:step_pid,1],title='Kp theta')
		fig,ax = plotter.plot(times[1:step_pid+1],kd[0:step_pid,0],title='Kd pos')
		fig,ax = plotter.plot(times[1:step_pid+1],kd[0:step_pid,1],title='Kd theta')

	# extract reference trajectory
	if param.controller_class in ['PID_wRef','Ref'] and not isinstance(pid_controller,ZeroPolicy):
		ref_state = extract_ref_state(pid_controller, states_pid)
		for i in range(env.n):
			fig,ax = plotter.plot(times[1:step_pid+1],ref_state[0:step_pid,i],title="ref " + env.states_name[i])

	# visualize
	if visualize:
		plotter.visualize(param, env, states_deeprl)

	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)
