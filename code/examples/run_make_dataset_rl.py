from run import run
from systems.cartpole import CartPole
from run_cartpole import CartpoleParam
from planning.rrt import rrt
from planning.scp import scp
import numpy as np
import sys, os
import torch

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

if __name__ == '__main__':
	param = CartpoleParam()
	param.env_case = 'Any90' #'SmallAngle','Swing90','Swing180', 'Any90'

	env = CartPole(param)

	deeprl_controller = torch.load(param.sim_rl_model_fn)
	times = param.sim_times

	for i in range(500):

		k = 0
		while True:
			param.scp_fn = '../models/CartPole/dataset_rl/scp_{}.csv'.format(k)
			param.scp_pdf_fn = '../models/CartPole/dataset_rl/scp_{}.pdf'.format(k)
			if not os.path.isfile(param.scp_fn):
				break
			k += 1
		print("Running on ", param.scp_fn)

		initial_state = env.reset()
		states_deeprl, actions_deeprl, step_deeprl = run_sim(deeprl_controller, initial_state)

		data = np.hstack([states_deeprl[0:-1], actions_deeprl])
		np.savetxt(param.rrt_fn, data, delimiter=',')

		scp(param, env)