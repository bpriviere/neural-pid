
# standard package
import torch
import gym 
from numpy import identity, zeros, array, vstack, pi, radians
import numpy as np 

# my package
import plotter 

def sim(param, env, visualize):

	def run_sim(controller, initial_state):
		states = zeros((len(times), env.n))
		actions = zeros((len(times) - 1, env.m))
		states[0] = env.reset(initial_state)
		reward = 0 
		for step, time in enumerate(times[:-1]):
			state = states[step]			
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
		return states, actions

	def extract_gains(controller, states):
		kp = zeros((len(times)-1,2))
		kd = zeros((len(times)-1,2))
		i = 0
		for state in states[1:]:
			kp[i] = controller.get_kp(state)
			kd[i] = controller.get_kd(state)
			i += 1
		return kp,kd

	def extract_ref_state(controller, states):
		ref_state = zeros((len(times)-1,4))
		for i, state in enumerate(states[1:]):
			ref_state[i] = controller.get_ref_state(state)
		return ref_state


	# -------------------------------------------

	# environment
	times = param.sim_times
	device = "cpu"

	# get controllers
	deeprl_controller = torch.load(param.sim_rl_model_fn)
	pid_controller = torch.load(param.sim_il_model_fn)
	# plain_pid_controller = PlainPID([2, 40], [4, 20])

	# run sim
	# s0 = array([-4,0,0,0])
	initial_state = env.reset()
	states_deeprl, actions_deeprl = run_sim(deeprl_controller, initial_state)
	states_pid, actions_pid = run_sim(pid_controller, initial_state)
	
	# states_pid = states_deeprl
	# actions_pid = actions_deeprl


	# plots
	for i in range(env.n):
		fig, ax = plotter.plot(times,states_deeprl[:,i],title=env.states_name[i])
		plotter.plot(times,states_pid[:,i], fig = fig, ax = ax)
	for i in range(env.m):
		fig, ax = plotter.plot(times[1:],actions_deeprl[:,i],title=env.actions_name[i])
		plotter.plot(times[1:],actions_pid[:,i], fig = fig, ax = ax)

	# extract gains
	if param.controller_class in ['PID_wRef','PID']:
		kp,kd = extract_gains(pid_controller,states_pid)
		fig,ax = plotter.plot(times[1:],kp[:,0],title='Kp pos')
		fig,ax = plotter.plot(times[1:],kp[:,1],title='Kp theta')
		fig,ax = plotter.plot(times[1:],kd[:,0],title='Kd pos')
		fig,ax = plotter.plot(times[1:],kd[:,1],title='Kd theta')

	# extract reference trajectory
	if param.controller_class in ['PID_wRef','Ref']:
		ref_state = extract_ref_state(pid_controller, states_pid)
		for i in range(env.n):
			fig,ax = plotter.plot(times[1:],ref_state[:,i],title="ref " + env.states_name[i])

	# visualize
	if visualize:
		plotter.visualize(env,states_deeprl)

	plotter.save_figs()
	plotter.open_figs()
