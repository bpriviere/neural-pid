
# standard package
import torch
import gym 
from numpy import identity, zeros, array, vstack

# my package
import plotter 
from systems import CartPole
from param import param 

def main():

	def run_sim(controller):
		states = zeros((len(times), env.n))
		actions = zeros((len(times) - 1, env.m))	
		states[0] = env.reset()
		reward = 0 
		for step, time in enumerate(times[:-1]):
			state = states[step]			
			action = controller.policy(state) 
			reward += env.reward()
			states[step + 1], _, done, _ = env.step(action)
			actions[step] = action
			if done:
				break
		print('reward: ',reward)
		env.close()
		return states, actions

	def extract_gains(controller, states):
		kp = zeros((len(times)-1,1))
		kd = zeros((len(times)-1,1))
		i = 0
		for state in states[1:]:
			kp[i] = controller.get_kp(state)
			kd[i] = controller.get_kd(state)
			i += 1
		return kp,kd

	def temp(controller,states):
		actions = zeros((len(times) - 1, env.m))	
		step = 0
		for state in states[:-1]:
			action = controller.policy(state)
			actions[step] = action
			step+=1
		return array(actions)

	# environment
	times = param.get('times')
	if param.get('system') is 'CartPole':
		env = CartPole()

	# get controllers
	deeprl_controller = torch.load(param.get('rl_model_fn'))
	pid_controller = torch.load(param.get('gains_model_fn'))

	# run sim 
	states_deeprl, actions_deeprl = run_sim(deeprl_controller)	
	# actions_pid = temp(pid_controller,states_deeprl)
	states_pid, actions_pid = run_sim(pid_controller)

	# extract gains
	kp,kd = extract_gains(pid_controller,states_pid)

	# plots
	for i in range(env.n):
		fig, ax = plotter.plot(times,states_deeprl[:,i],title=param.get('states_name')[i])
		plotter.plot(times,states_pid[:,i], fig = fig, ax = ax)
	for i in range(env.m):
		fig, ax = plotter.plot(times[1:],actions_deeprl[:,i],title=param.get('actions_name')[i])
		plotter.plot(times[1:],actions_pid[:,i], fig = fig, ax = ax)

	fig,ax = plotter.plot(times[1:],kp,title='Kp')
	fig,ax = plotter.plot(times[1:],kd,title='Kd')


	plotter.save_figs()
	plotter.open_figs()

if __name__ == '__main__':
	main()