
# standard package
import torch
import gym 
from numpy import identity, zeros, array, vstack, pi, radians
import argparse

# my package
import plotter 
from systems import CartPole
from param import param 
from learning import PlainPID

def main(visualize):

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

	def compute_actions(controller, states):
		actions = zeros((len(times) - 1, env.m))
		for step, time in enumerate(times[:-1]):
			state = states[step]			
			action = controller.policy(state) 
			actions[step] = action
		return actions

	def extract_gains(controller, states):
		kp = zeros((len(times)-1,2))
		kd = zeros((len(times)-1,2))
		i = 0
		for state in states[1:]:
			kp[i] = controller.get_kp(state)
			kd[i] = controller.get_kd(state)
			i += 1
		return kp,kd

	def extract_ref_state(controller, states):
		ref_state = zeros((len(times)-1,4))
		for i, state in enumerate(states[1:]):
			ref_state[i] = controller.get_ref_state(state)
		return ref_state

	def temp(controller,states):
		actions = zeros((len(times) - 1, env.m))	
		step = 0
		for state in states[:-1]:
			action = controller.policy(state)
			actions[step] = action
			step+=1
		return array(actions)

	# environment
	times = param.sim_times
	if param.env_name is 'CartPole':
		env = CartPole()

	# get controllers
	deeprl_controller = torch.load(param.rl_model_fn)
	pid_controller = torch.load(param.il_model_fn)
	plain_pid_controller = PlainPID([2, 40], [4, 20])

	# run sim
	initial_state = env.reset()
	# initial_state = [1, radians(5), 0, 0.5]
	states_deeprl, actions_deeprl = run_sim(deeprl_controller, initial_state)
	# actions_pid = temp(pid_controller,states_deeprl)
	states_pid, actions_pid = run_sim(pid_controller, initial_state)

	# states_pid = states_deeprl
	# actions_pid = compute_actions(pid_controller, states_pid)

	# stated_plain_pid, actions_plain_pid = run_sim(deeprl_controller, initial_state)

	# extract gains
	kp,kd = extract_gains(pid_controller,states_pid)
	# ref_state = extract_ref_state(pid_controller, states_pid)

	# plots
	for i in range(env.n):
		fig, ax = plotter.plot(times,states_deeprl[:,i],title=env.states_name[i])
		plotter.plot(times,states_pid[:,i], fig = fig, ax = ax)
		# plotter.plot(times,stated_plain_pid[:,i], fig = fig, ax = ax)
	for i in range(env.m):
		fig, ax = plotter.plot(times[1:],actions_deeprl[:,i],title=env.actions_name[i])
		plotter.plot(times[1:],actions_pid[:,i], fig = fig, ax = ax)
		# plotter.plot(times[1:],actions_plain_pid[:,i], fig = fig, ax = ax)

	fig,ax = plotter.plot(times[1:],kp[:,0],title='Kp pos')
	fig,ax = plotter.plot(times[1:],kp[:,1],title='Kp theta')
	fig,ax = plotter.plot(times[1:],kd[:,0],title='Kd pos')
	fig,ax = plotter.plot(times[1:],kd[:,1],title='Kd theta')

	# for i in range(env.n):
	# 	fig,ax = plotter.plot(times[1:],ref_state[:,i],title="ref " + env.states_name[i])


	plotter.save_figs()
	plotter.open_figs()

	# visualize 3D
	if visualize:
		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		import time

		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		vis["cart"].set_object(g.Box([0.5,0.2,0.2]))
		vis["pole"].set_object(g.Cylinder(env.length_pole, 0.01))

		while True:
			for t, state in zip(times, states_deeprl):
				vis["cart"].set_transform(tf.translation_matrix([state[0], 0, 0]))

				vis["pole"].set_transform(
					tf.translation_matrix([state[0], 0, env.length_pole/2]).dot(
						tf.euler_matrix(pi/2, state[1], 0)))

				time.sleep(0.1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--animate", action='store_true')
	args = parser.parse_args()
	main(args.animate)