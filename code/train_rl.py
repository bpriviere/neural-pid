
# from continuous_ppo import PPO, Memory
from learning import PPO, PPO_c
from systems import CartPole
from param import param 
import torch 
from torch.distributions import MultivariateNormal,Categorical
import numpy as np 

def main():

	continuous = param.rl_continuous_on
	if param.rl_continuous_on:
		print('Continuous Action Space')
	else:
		print('Discrete Action Space: ',param.rl_discrete_action_space)
	print("Case: ", param.env_case)

	# creating environment
	env_name = 'CartPole'
	env = CartPole()
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
	solved_reward = 0.97*env.max_reward()*param.sim_nt
	print('Solved Reward: ',solved_reward)

	# init model 
	if continuous:
		model = PPO_c(state_dim,action_dim,param.rl_action_std,param.rl_cuda_on,param.rl_lr, 
			param.rl_gamma, param.rl_K_epoch, param.rl_lmbda, param.rl_eps_clip)
	else:
		model = PPO(param.rl_discrete_action_space, state_dim,action_dim,param.rl_action_std,param.rl_cuda_on,param.rl_lr, 
			param.rl_gamma, param.rl_K_epoch, param.rl_lmbda, param.rl_eps_clip)

	# logging variables
	running_reward = 0
	count = 0
	best_reward = 0 
	
	# training loop
	for i_episode in range(1, param.rl_max_episodes+1):
		while len(model.data) <= param.rl_ndata_per_epi:
			s = env.reset()
			done = False
			count += 1.
			for step, time in enumerate(times[:-1]):
				if continuous:
					a,log_prob_a = model.policy(s)
					s_prime, r, done, _ = env.step(a)
					model.put_data((s,a,r,s_prime,log_prob_a,done))
				else:
					prob = model.pi(torch.from_numpy(s).float())
					classification = Categorical(prob).sample().item()
					action = model.class_to_force(classification)
					s_prime, r, done, _ = env.step([action])
					model.put_data((s,classification,r,s_prime,prob[classification].item(),done))
				s = s_prime
				running_reward += r
				if done:
					break

		model.train_net()
		
		# stop training if avg_reward > solved_reward
		if running_reward/count > solved_reward:
			print("########## Solved! ##########")
			torch.save(model, param.rl_train_model_fn)
			break
					
		# logging
		if i_episode % param.rl_log_interval == 0:			
			print('Episode {} \t Avg reward: {:2f}'.format(i_episode, running_reward/count))
			
			# save best iteration
			if running_reward/count > best_reward:
				best_reward = running_reward/count
				print('   saving best model')
				torch.save(model, param.rl_train_model_fn)

			running_reward = 0
			count = 0 



if __name__ == '__main__':
	main()