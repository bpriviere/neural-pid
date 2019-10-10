

# standard package
import torch
import gym 
from numpy import identity, zeros, array, vstack, pi, radians
import argparse
import matplotlib.pyplot as plt 

# my package
import plotter 
from systems import CartPole
from learning import PlainPID


def main():


	def run_sim(controller, initial_state):
		states = zeros((len(times), env.n))
		actions = zeros((len(times) - 1, env.m))
		states[0] = env.reset(initial_state)
		reward = 0 
		for step, time in enumerate(times[:-1]):
			state = states[step]			
			action = controller.policy(state) 
			reward += env.reward()
			states[step + 1], _, done, _ = env.step([action])
			actions[step] = action
			if done:
				break
		print('reward: ',reward)
		env.close()
		return states, actions

	pid_controllers = [
		'woRef','wRef','Ref','Plain']
	env_cases = [
		'SmallAngle','Swing90']
	
	# environment
	times = param.sim_times
	n_cases = len(pid_controllers)*len(env_cases)
	plot_states = zeros((len(times), n_cases*2))

	plot_case_count = 0
	for env_case in env_cases:
		for pid_controller in pid_controllers:
			

			param.env_case = env_case
			if param.env_name is 'CartPole':
				env = CartPole()

			if pid_controller == 'Plain':
				model = PlainPID([2, 40], [4, 20])
			else:
				model_fn = '../models/il_model' + '_' + \
					pid_controller + '_' + env_case + '.pt'
				print(model_fn)

				model = torch.load(model_fn)

			initial_state = env.reset()
			states,actions = run_sim(model, initial_state)
			
			plot_states[:,plot_case_count] = states[:,0]
			plot_states[:,plot_case_count+1] = states[:,1]
			plot_case_count += 2


	fig,ax = plotter.plot(times,plot_states[:,0],title=env.states_name[0],label=pid_controllers[0])
	plotter.plot(times,plot_states[:,2],fig=fig,ax=ax,label=pid_controllers[1])
	plotter.plot(times,plot_states[:,4],fig=fig,ax=ax,label=pid_controllers[2])
	plotter.plot(times,plot_states[:,6],fig=fig,ax=ax,label=pid_controllers[3])

	fig,ax = plotter.plot(times,plot_states[:,1],title=env.states_name[1],label=pid_controllers[0])
	plotter.plot(times,plot_states[:,3],fig=fig,ax=ax,label=pid_controllers[1])
	plotter.plot(times,plot_states[:,5],fig=fig,ax=ax,label=pid_controllers[2])
	plotter.plot(times,plot_states[:,7],fig=fig,ax=ax,label=pid_controllers[3])

	fig,ax = plotter.plot(times,plot_states[:,8],title=env.states_name[0],label=pid_controllers[0])
	plotter.plot(times,plot_states[:,10],fig=fig,ax=ax,label=pid_controllers[1])
	plotter.plot(times,plot_states[:,12],fig=fig,ax=ax,label=pid_controllers[2])
	plotter.plot(times,plot_states[:,14],fig=fig,ax=ax,label=pid_controllers[3])

	fig,ax = plotter.plot(times,plot_states[:,9],title=env.states_name[1],label=pid_controllers[0])
	plotter.plot(times,plot_states[:,11],fig=fig,ax=ax,label=pid_controllers[1])
	plotter.plot(times,plot_states[:,13],fig=fig,ax=ax,label=pid_controllers[2])
	plotter.plot(times,plot_states[:,15],fig=fig,ax=ax,label=pid_controllers[3])

	plotter.save_figs()
	plotter.open_figs()


if __name__ == '__main__':
	main()