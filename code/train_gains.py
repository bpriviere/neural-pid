import torch
import torch.nn.functional as F
import torch.utils.data as Data

from numpy import array
from numpy.random import uniform,seed
from torch.distributions import Categorical

from param import param 
from learning import PIDNet, GainsNet

def make_dataset():
	model = torch.load(param.get('rl_model_fn'))
	states = []
	actions = []
	for _ in range(param.get('gains_n_data')):
		state = array((
			param.get('sys_pos_bounds')*uniform(-1.,1.),
			param.get('sys_angle_bounds_deg')*uniform(-1.,1.),
			2.*uniform(-1.,1.),
			2.*uniform(-1.,1.),         
			))
		action = model.policy(state)
		action = action.reshape((-1))
		states.append(state)
		actions.append(action)

	return torch.tensor(states).float(), torch.tensor(actions).float()

def train(model, loader):

	optimizer = torch.optim.Adam(model.parameters(), lr=param.get('gains_lr'))
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

	# init model
	model = PIDNet()

	# datasets
	x_train,y_train = make_dataset() 
	dataset_train = Data.TensorDataset(x_train, y_train)
	loader_train = Data.DataLoader(
		dataset=dataset_train, 
		batch_size=param.get('gains_batch_size'), 
		shuffle=True)
	x_test,y_test = make_dataset() 
	dataset_test = Data.TensorDataset(x_test, y_test)
	loader_test = Data.DataLoader(
		dataset=dataset_test, 
		batch_size=param.get('gains_batch_size'), 
		shuffle=True)

	for epoch in range(param.get('gains_n_epoch')):
		pre_test_epoch_loss = test(model, loader_test)
		train_epoch_loss = train(model,loader_train)
		post_test_epoch_loss = test(model, loader_test)
		
		if epoch%param.get('gains_print_epoch')==0:
			print('epoch: ', epoch)
			print('   Pre Test Epoch Loss: ', pre_test_epoch_loss)
			print('   Train Epoch Loss: ', train_epoch_loss)
			print('   Post Test Epoch Loss: ', post_test_epoch_loss)

	torch.save(model, param.get('gains_model_fn'))

if __name__ == '__main__':
	main()
