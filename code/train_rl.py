
# standard packages
import gym
import torch
from torch.distributions import Categorical
from numpy import array, vstack, zeros

# my packages
from systems import CartPole
from param import param 
from learning import PPO 


def main():
	
	times = param.get('times')
	
	# environment
	if param.get('system') is 'CartPole':
		env = CartPole()
	
	# model
	model = PPO()

	# train
	score = 0.0
	print_interval = 20
	best_score = 0
	ave_score_break = 0.99*len(times)*env.max_reward()	
	if env.objective is 'track':
		ave_score_break = 0.95*ave_score_break

	for n_epi in range(param.get('rl_n_eps')):
		s = env.reset()
		done = False
		for step, time in enumerate(times[:-1]):
			prob = model.pi(torch.from_numpy(s).float())
			classification = Categorical(prob).sample().item()
			action = model.class_to_force(classification)
			s_prime, r, done, info = env.step(action)
			model.put_data((s, classification, r, s_prime, prob[classification].item(), done))
			s = s_prime
			score += r
			if done:
			    break

		model.train_net()

		if n_epi%print_interval==0 and n_epi!=0:
			print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
			# if score/print_interval >= ave_score_break:
			# 	break
			if score > best_score:
				best_score = score
				torch.save(model, param.get('rl_model_fn'))
			score = 0.0

	torch.save(model, param.get('rl_model_fn'))
	env.close()

if __name__ == '__main__':
    main()