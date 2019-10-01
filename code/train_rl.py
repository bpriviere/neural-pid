
# standard packages
import gym
import torch
from torch.distributions import Categorical
from numpy import array, vstack, zeros
import time 

# my packages
from systems import CartPole
from param import param 
from learning import PPO 
from utilities import eval_normal_prob

def main():
	
	times = param.get('times')
	
	# environment
	if param.get('system') is 'CartPole':
		env = CartPole()
	
	# model
	model = PPO_c()

	# train
	score = 0.0
	print_interval = 20
	best_score = 0

	for n_epi in range(param.get('rl_n_eps')):
		s = env.reset()
		done = False
		for step, time in enumerate(times[:-1]):
			prob = model.pi(torch.from_numpy(s).float())
			action = prob.sample()
			log_prob_a = prob.log_prob(action)
			s_prime, r, done, info = env.step(action)
			model.put_data((s, action, r, s_prime, 
				log_prob_a, done))
			s = s_prime
			score += r

			# print(prob)
			# print(action)
			# print(prob_a)
			# print(s)
			# exit()
			if done:
			    break

		model.train_net()

		if n_epi%print_interval==0 and n_epi!=0:
			print("# of episode :{}, avg score : {:.3f}".format(n_epi, score/print_interval))
			if score > best_score:
				best_score = score
				torch.save(model, param.get('rl_model_fn'))
			score = 0.0

	torch.save(model, param.get('rl_model_fn'))
	env.close()

if __name__ == '__main__':
	t = time.time()
	main()
	elapsed = time.time() - t
	print('RL Training Time:', elapsed)