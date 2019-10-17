
# standard package
import torch
import gym 
import numpy as np 
import os

# my package
import plotter 

def main():

	def run_sim(controller, initial_state):
		states = np.zeros((len(times), env.n))
		actions = np.zeros((len(times) - 1, env.m))
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
		return states, actions, step


	def agent_idx_to_agent_state(s,i):
		pass 

	# -------------------------------------------

	# environment

	# get controllers
	deeprl_controller = torch.load('temp.pt')
	
	# run sim
	s0 = np.array([0,50,0,0,0,40.4509,29.3893,0,0,15.4509,47.5528,0,0,-15.4509,47.5528,0,0,-40.4509,29.3893,0,0,-50,6.12323e-15,0,0,-40.4509,-29.3893,0,0,-15.4509,-47.5528,0,0,15.4509,-47.5528,0,0,40.4509,-29.3893,0,0])
	print(s0.shape)
	exit()
	num_agents = 10 


	

if __name__ == '__main__':
	main()