
# standard package
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from numpy import zeros, array, dot 
from numpy.random import uniform 
from torch.distributions import Categorical

# my stuff
from param import param
from learning import GainsNet, GainsNet2


def make_dataset():
	
	data = []
	model = torch.load(param.get('rl_model_fn'))

	for _ in range(param.get('gains_n_batch')):
		states = []
		actions = []
		for _ in range(param.get('gains_n_data')):
			state = array((
				param.get('sys_pos_bounds')*uniform(-1.,1.),
				param.get('sys_angle_bounds_deg')*uniform(-1.,1.),
				2.*uniform(-1.,1.),
				2.*uniform(-1.,1.),			
				))
			prob = model.pi(torch.from_numpy(state).float())
			m = Categorical(prob)
			action = array(param.get('sys_actions')[m.sample().item()],ndmin=1)

			states.append(state)
			actions.append(action)
		data.append(array((states,actions)))
	return data

def train(model,dataset):

	loss_func = torch.nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr = param.get('gains_lr'))

	for batch_idx, (data, target) in enumerate(dataset):
		data = torch.from_numpy(data).float()
		target = torch.from_numpy(target).float()
		# prediction = model.calc_action(data)
		prediction = model(data)
		loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)

		# print(loss.grad_fn)
		# print(loss.grad_fn.next_functions[0][0])
		# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # apply gradients

def test(model,dataset):

	Loss = 0
	loss_func = torch.nn.MSELoss()
	for batch_idx, (data, target) in enumerate(dataset):
		data = torch.from_numpy(data).float()
		target = torch.from_numpy(target).float()
		# prediction = model.calc_action(data)
		prediction = model(data)
		loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)
		
		if batch_idx % param.get('gains_log_interval') == 0:
			print('   loss: ', loss)

	# print('Test Loss: ', Loss)


def main():

	device = "cpu"
	model = GainsNet2()

	train_dataset = make_dataset()
	test_dataset = make_dataset()

	for epoch in range(param.get('gains_epochs')):
		print('Epoch: ', epoch + 1)
		train(model, train_dataset)
		test(model, test_dataset)

	torch.save(model,param.get('gains_model_fn'))


if __name__ == '__main__':
	main()