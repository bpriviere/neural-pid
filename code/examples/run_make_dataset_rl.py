from run import run
from systems.cartpole import CartPole
from systems.quadrotor import Quadrotor
from run_cartpole import CartpoleParam
from run_quadrotor import QuadrotorParam
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
		states[step + 1], r, done, _ = env.step(action)
		reward += r
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

	# param = QuadrotorParam()
	# env = Quadrotor(param)

	# param.sim_rl_model_fn = '../models/quadrotor/rl_discrete.pt'

	deeprl_controller = torch.load(param.sim_rl_model_fn)
	times = param.sim_times
	xf = param.ref_trajectory[:,-1]

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
		x, u, step = run_sim(deeprl_controller, initial_state)
		x = x[0:-1]
		if step + 2 == len(times):
			try:
				# x, u, obj = scp(param, env, x, u, "minimizeError")
				# if obj < 1e-6:
				x, u, obj = scp(param, env, x, u, "minimizeX")
				idx = -1
				# error = np.linalg.norm(x - xf, axis=1)
				# print(error)
				# idx = np.where(error<0.5)
				# if len(idx) > 0:
				# 	idx = idx[0][0]
				# 	print(idx)
				result = np.hstack([x[0:idx], u[0:idx]])
				np.savetxt(param.scp_fn, result, delimiter=',')
			except Exception as e:
				print(e)
				print("Error during SCP. Skipping.")
				pass