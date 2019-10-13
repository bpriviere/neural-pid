
# my packages
from learning.ddpg import DDPG

# standard packages
import torch 
from torch.distributions import MultivariateNormal,Categorical
import numpy as np 

import torch.optim as optim
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test




def autotune(param,env):

	def tune_train(config):

		continuous = param.rl_continuous_on

		state_dim = env.n
		action_dim = env.m
		times = param.sim_times

		# init model 
		if continuous:
			model = DDPG(
				state_dim,
				action_dim,
				param.rl_control_lim,
				config["lr_mu"],
				config["lr_q"],
				param.rl_gamma,
				param.rl_batch_size,
				param.rl_buffer_limit,
				param.rl_action_std,
				param.rl_tau,
				param.rl_K_epoch,
				param.rl_max_action_perturb,
				param.rl_gpu_on)
		else:
			model = PPO(
				param.rl_discrete_action_space, 
				state_dim,
				action_dim,
				param.rl_action_std,
				param.rl_cuda_on,
				param.rl_lr, 
				param.rl_gamma, 
				param.rl_K_epoch, 
				param.rl_lmbda, 
				param.rl_eps_clip)

		# logging variables
		trial_count = 0
		best_reward = -np.Inf
		data_count = 0
		running_reward = 0

		# training loop
		for i_episode in range(1, param.rl_max_episodes+1):
			while data_count <= param.rl_batch_size:
				s = env.reset()
				done = False
				trial_count += 1.
				for step, time in enumerate(times[:-1]):

					if continuous:
						a = model.train_policy(s)
						s_prime, r, done, _ = env.step(a)
						model.put_data((s,a,r,s_prime,done))

					else:
						prob = model.pi(torch.from_numpy(s).float())
						c = Categorical(prob).sample().item()
						a = model.class_to_force(c)
						s_prime, r, done, _ = env.step([a])
						model.put_data((s,c,r,s_prime,prob[c].item(),done))

					s = s_prime
					running_reward += r
					data_count += 1
					if done:
						break

			model.train_net()
			data_count = 0
			ave_reward = running_reward/trial_count
			tune.track.log(mean_accuracy=ave_reward)

	config = {
		"lr_mu": tune.grid_search([1e-5,1e-4,1e-3,1e-2]),
		"lr_q": tune.grid_search([1e-5,1e-4,1e-3,1e-2]),
	}

	analysis = tune.run(
	    tune_train, config=config)

	print("Best config: ", analysis.get_best_config(metric="ave_reward"))

	# Get a dataframe for analyzing trial results.
	df = analysis.dataframe()

