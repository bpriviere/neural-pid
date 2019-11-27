
# my packages
from learning.ppo_v2 import PPO
from learning.ddpg import DDPG
from learning.ppo_w_deepset import PPO_w_DeepSet
from utilities import debug_lst
import plotter

# standard packages
import torch 
from torch.distributions import MultivariateNormal,Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np 

class ReduceLROnRewardSchedule:
	def __init__(self,optimizer,gamma,schedule):
		self.optimizer = optimizer
		self.initial_lr = optimizer.param_groups[0]['lr']
		self.gamma = gamma
		self.schedule = schedule
		# print('reward schedule: ', self.reward_schedule)

	def step(self,reward):
		k = np.where(reward > self.schedule)[0][-1]
		for param_group in self.optimizer.param_groups:
			lr = self.initial_lr*(self.gamma**k)
			if param_group['lr'] > lr:
				print("Changing Learning Rate From: %5f to %5f"%((param_group['lr'],lr)))
				param_group['lr'] = lr


def train_rl(param, env):

	continuous = param.rl_continuous_on
	if param.rl_continuous_on:
		print('Continuous Action Space')
	else:
		print('Discrete Action Space: ',param.rl_discrete_action_space)
	print("Case: ", param.env_case)

	state_dim = env.n
	action_dim = env.m
	times = param.sim_times

	# random seed
	random_seed = 1
	if random_seed:
		print("Random Seed: {}".format(random_seed))
		torch.manual_seed(random_seed)
		env.seed(random_seed)
		np.random.seed(random_seed)

	# exit condition
	solved_reward = 0.97*env.max_reward*param.sim_nt
	print('Solved Reward: ',solved_reward)

	# init model 
	if param.rl_warm_start_on:
		print('Loading Previous Model: ', param.rl_warm_start_fn)
		model = torch.load(param.rl_warm_start_fn)
		if continuous:
			model.make_replay_buffer(int(param.rl_buffer_limit))
		print(model)
	else:
		print('Creating New Model...')
		print(param)
		if param.rl_module is 'DDPG':
			print('DDPG')
			model = DDPG(
				param.rl_mu_network_architecture,
				param.rl_q_network_architecture,
				param.rl_network_activation,
				param.a_min,
				param.a_max,
				param.rl_action_std,
				param.rl_max_action_perturb,
				param.rl_lr_mu,
				param.rl_lr_q,
				param.rl_tau,
				param.rl_gamma,
				param.rl_batch_size,
				param.rl_K_epoch,
				param.rl_buffer_limit,
				param.rl_gpu_on)

		elif param.rl_module is 'PPO':
			print('PPO')
			model = PPO(
				param.rl_discrete_action_space, 
				state_dim,
				action_dim,
				param.rl_layers, 
				param.rl_activation,
				param.rl_gpu_on,
				param.rl_lr, 
				param.rl_gamma, 
				param.rl_K_epoch, 
				param.rl_lmbda, 
				param.rl_eps_clip)

	if param.rl_lr_schedule_on:
		schedulers = []
		for optimizer in model.get_optimizers():
			schedulers.append(ReduceLROnRewardSchedule(optimizer,\
				param.rl_lr_schedule_gamma,
				param.rl_lr_schedule))

	# logging variables
	running_reward = 0
	trial_count_per_interval = 0
	trial_count_per_batch = 0
	best_reward = -np.Inf
	data_count = 0

	# training loop
	for i_episode in range(1, param.rl_max_episodes+1):
		while data_count <= param.rl_batch_size:
			s = env.reset()
			done = False
			trial_count_per_batch += 1.

			for step, time in enumerate(times[:-1]):

				if param.rl_module is 'DDPG':
					a = model.train_policy(s)
					s_prime, r, done, _ = env.step(a)
					model.put_data((s,a,r,s_prime,done))

				elif param.rl_module is 'PPO' and param.pomdp_on:
					observations = env.observe() 
					observations = env.unpack_observations(observations)
					
					o_lst = []
					c_lst = []
					r_lst = []
					p_lst = []
					d_lst = []
					a_lst = []
					
					# for debugging not used by learning 
					s_lst = []
					sp_lst = []

					# try training with data from one agent: 
					if True: 
						for i_good_node in range(env.n_agents):
							if env.good_nodes[i_good_node]:
								break

						training_data_agents = [env.agents[i_good_node]]
						
					# try training with data from all agents 
					else:
						training_data_agents = []
						for agent in env.agents:
							if env.good_nodes[agent.i]:
								training_data_agents.append(agent)
						
					for agent_i in env.agents:
						o_i = torch.tensor(observations[agent_i.i]).float()
						c_i = Categorical(model.pi(o_i)).sample().item()
						p_i = model.pi(o_i)[c_i].item()
						a_i = model.class_to_action(c_i)
						s_i = agent_i.x
						sp_i, r_i, d_i = env.step_i(agent_i, a_i)

						# print('i: ', agent_i.i)
						# print('o_i: ', o_i)
						# print('c_i: ', c_i)
						# print('p_i: ', p_i)
						# print('a_i: ', a_i)
						# print('r_i: ', r_i)
						# print('d_i: ', d_i)
						# print('s_i: ', s_i)
						# print('sp_i: ', sp_i)
						# exit()
					
						o_lst.append(o_i.numpy())
						c_lst.append(c_i)
						r_lst.append(r_i)
						p_lst.append(p_i)
						d_lst.append(d_i)
						a_lst.append(a_i)

						# for debugging 
						s_lst.append(s_i)
						sp_lst.append(sp_i)
						
					s_prime, r, d, _ = env.step(a_lst)
					op_lst = env.observe(update_agents=False)
					op_lst = env.unpack_observations(op_lst)

					for agent_i in training_data_agents:
						i = agent_i.i


						# print('putting data')
						# print('i: ', i)
						# print('state_i: ', s_lst[agent_i.i])
						# print('observation_i: ', observations[agent_i.i])
						# print('unpacked observation_i: ', unpacked_observations[agent_i.i])
						# print('o_i: ', o_lst[i])
						# exit()

						model.put_data((np.array(o_lst[i]),c_lst[i],
							r_lst[i],np.array(op_lst[i]),
							p_lst[i],d_lst[i]))

						# print('making batch:')
						# print('	i: ', i)
						# print('	o_lst[i]: ', o_lst[i])
						# print('c_lst[i]: ', c_lst[i])
						# print('r_lst[i]: ', r_lst[i])
						# print('	op_lst[i]: ', op_lst[i])
						# print('p_lst[i]: ', p_lst[i])
						# print('d_lst[i]: ', d_lst[i])
						# print('	s_lst: ', s_lst)
						# print('	sp_lst: ', sp_lst)
						# if step == 2:
						# 	exit()

				elif param.rl_module is 'PPO' and not param.pomdp_on:
					prob = model.pi(torch.from_numpy(s).float())
					c = Categorical(prob).sample().item()
					a = model.class_to_action(c)
					s_prime, r, done, _ = env.step([a])
					model.put_data((s,c,r,s_prime,prob[c].item(),done))
					
				s = s_prime
				running_reward += r
				
				data_count += 1
				# data_count += env.n_agents - env.n_malicious

				if done:
					break


		
		trial_count_per_interval += trial_count_per_batch
		# print('trial_count_per_batch: ', trial_count_per_batch)
		# print('trial_count_per_interval: ', trial_count_per_interval)
		model.train_net()
		data_count = 0
		trial_count_per_batch = 0
		
		
		# stop training if avg_reward > solved_reward
		# if running_reward/trial_count_per_interval > solved_reward:
		# 	print('Episode {} \t Avg reward: {:2f}'.format(i_episode, running_reward/trial_count_per_interval))
		# 	print("########## Solved! ##########")
		# 	temp_buffer = model.data
		# 	model.data = []
		# 	torch.save(model, param.rl_train_model_fn)
		# 	model.data = temp_buffer
		# 	break
					
		# logging
		if i_episode % param.rl_log_interval == 0:			
			print('Episode {} \t Avg reward: {:2f}'.format(i_episode, running_reward/trial_count_per_interval))
			
			# save latest
			temp_buffer = model.data
			model.data = []
			torch.save(model, param.rl_train_latest_model_fn)
			model.data = temp_buffer


			# save best iteration
			if running_reward/trial_count_per_interval > best_reward:
				best_reward = running_reward/trial_count_per_interval
				print('   saving best model')
				temp_buffer = model.data
				model.data = []
				torch.save(model, param.rl_train_best_model_fn)
				model.data = temp_buffer

			# update learning rate
			if param.rl_lr_schedule_on:
				for scheduler in schedulers:
					scheduler.step(running_reward/trial_count_per_interval)

			running_reward = 0
			trial_count_per_interval = 0 