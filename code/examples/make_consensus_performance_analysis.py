
# my package
from param import Param
from run import run
from systems.consensus import Consensus
from run_consensus import ConsensusParam
from other_policy import LCP_Policy, WMSR_Policy
from learning.ppo_v2 import PPO
import plotter 

# standard package
import numpy as np 
from torch import nn as nn 
from torch import tanh
from torch.nn import LeakyReLU
import torch 
import os 


def run_sim(controller, initial_state):
	states = np.zeros((len(times), env.n))
	actions = np.zeros((len(times)-1,env.m))
	rewards = np.zeros((len(times)-1))
	observations = [] 
	total_reward = 0 

	env.reset(initial_state)
	states[0] = np.copy(env.state)
	for step, time in enumerate(times[:-1]):
		state = states[step]
		observation = env.observe()

		if param.env_name is 'Consensus' and (isinstance(controller, LCP_Policy) or isinstance(controller,PPO)):
			observation = env.unpack_observations(observation)

		
		action = controller.policy(observation) 
		next_state, r, done, _ = env.step(action)
		total_reward += r
		
		states[step + 1] = next_state
		actions[step] = action.flatten()
		rewards[step] = r
		observations.append(observation)

		if done:
			break

	#print('total reward: ',total_reward)
	env.close()
	return states, observations, rewards, actions, step



if __name__ == '__main__':
	param = ConsensusParam()
	env = Consensus(param)

	controllers = {
		'LCP': LCP_Policy(env),
		'WMSR': WMSR_Policy(env),
		#'RL':	torch.load(param.sim_rl_model_fn),
		'aLCP':	torch.load(param.sim_il_model_fn),
	}

	n_trials = 100
	alpha = 0.5
	times = param.sim_times

	fig,ax = plotter.make_fig()

	for name, controller in controllers.items():
		print('Controller: ', name)
		
		controller_rewards = np.empty((n_trials, len(times)-1))

		for i_trial in range(n_trials):
			print('   trial: ', i_trial/n_trials)

			x0 = env.reset()
			while not env.bad_nodes[2]:
				x0 = env.reset()
			_, _, controller_rewards[i_trial,:], _, step = run_sim(controller,x0)

		mean = np.mean(controller_rewards, axis=0)
		std = 0.5*np.std(controller_rewards, axis=0)

		l1 = ax.plot(times[1:], mean,label=name)
		
		color = l1[0].get_color()

		lower = mean-std
		upper = mean+std

		ax.fill_between(times[1:], lower, upper, alpha=alpha, color=color)

	ax.legend(loc='lower left')
	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)
