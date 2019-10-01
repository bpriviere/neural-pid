
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from numpy import array
from numpy.random import uniform,seed
from torch.distributions import Categorical

from param import param 
from learning import PIDNet,PIDNet_wRef,PlainPID
from systems import CartPole 

# def make_dataset(env):
# 	# model = PlainPID([2, 40], [4, 20])
# 	model = torch.load(param.rl_model_fn)
# 	states = []
# 	actions = []
# 	for _ in range(param.il_n_data):
# 		state = array((
# 			env.env_state_bounds[0]*uniform(-1.,1.),
# 			env.env_state_bounds[1]*uniform(-1.,1.),
# 			env.env_state_bounds[2]*uniform(-1.,1.),
# 			env.env_state_bounds[3]*uniform(-1.,1.),         
# 			))
# 		action = model.policy(state)
# 		action = action.reshape((-1))
# 		states.append(state)
# 		actions.append(action)
# 	return torch.tensor(states).float(), torch.tensor(actions).float()

def make_dataset(env):
	model = torch.load(param.rl_model_fn)
	times = param.sim_times
	states = []
	actions = []
	while len(states) < param.il_n_data:
		states.append(env.reset())
		action = model.policy(states[-1])
		actions.append(action.reshape((-1)))
		for step, time in enumerate(times[:-1]):
			action = model.policy(states[-1])
			s_prime, _, done, _ = env.step([action])
			states.append(s_prime)
			actions.append(action.reshape((-1)))
			if done:
				break

	states = states[0:param.il_n_data]
	actions = actions[0:param.il_n_data]			
	return torch.tensor(states).float(),torch.tensor(actions).float()


def train(model, loader):

	optimizer = torch.optim.Adam(model.parameters(), lr=param.il_lr)
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step
		prediction = model(b_x)     # input x and predict based on x
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # apply gradients
		epoch_loss += loss 
	return epoch_loss/step


def test(model, loader):
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step
		prediction = model(b_x)     # input x and predict based on x
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		epoch_loss += loss 
	return epoch_loss/step

def main():

	seed(1) # numpy random gen seed 
	torch.manual_seed(1)    # pytorch 

	# env
	if param.env_name is 'CartPole':
		env = CartPole()

	# init model
	model = PIDNet(env.n)

	# datasets
	x_train,y_train = make_dataset(env) 
	dataset_train = Data.TensorDataset(x_train, y_train)
	loader_train = Data.DataLoader(
		dataset=dataset_train, 
		batch_size=param.il_batch_size, 
		shuffle=True)
	x_test,y_test = make_dataset(env) 
	dataset_test = Data.TensorDataset(x_test, y_test)
	loader_test = Data.DataLoader(
		dataset=dataset_test, 
		batch_size=param.il_batch_size, 
		shuffle=True)

	for epoch in range(1,param.il_n_epoch+1):
		train_epoch_loss = train(model,loader_train)
		post_test_epoch_loss = test(model, loader_test)
		if epoch%param.il_log_interval==0:
			print('epoch: ', epoch)
			print('   Train Epoch Loss: ', train_epoch_loss)
			print('   Test Epoch Loss: ', post_test_epoch_loss)

	torch.save(model, param.il_model_fn)

if __name__ == '__main__':
	main()
