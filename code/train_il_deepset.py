
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn

import numpy as np

neighborDist = 15
nNeighbor = 10 


def load_dataset(filename):
	data = np.load(filename)
	num_agents = int((data.shape[1] - 1) / 4)
	# loop over each agent and each timestep to find
	#  * current state
	#  * set of neighboring agents (storing their relative states)
	#  * label (i.e., desired control)
	dataset = []
	for t in range(data.shape[0]-1):
		for i in range(num_agents):
			state_i = data[t,i*4+1:i*4+5]
			neighbors = []
			for j in range(num_agents):
				if i != j:
					state_j = data[t,j*4+1:j*4+5]
					dist = np.linalg.norm(state_i[0:2] - state_j[0:2])
					if dist <= neighborDist:
						neighbors.append(state_i - state_j)
			# desired control is the velocity in the next timestep
			u = data[t+1, i*4+3:i*4+5]
			dataset.append([state_i, neighbors, u])
	print(len(dataset))
	return dataset


def load_dataset_v2(filename):
	data = np.load(filename)
	num_agents = int((data.shape[1] - 1) / 4)
	# loop over each agent and each timestep to find
	#  * current state
	#  * set of neighboring agents (storing their relative states)
	#  * label (i.e., desired control)
	
	nt = data.shape[0]-1
	# nt = 6000

	X = np.nan*np.ones((nt,4*num_agents))
	Y = np.nan*np.ones((nt,2))

	for t in range(nt):
		for i in range(num_agents):
			state_i = data[t,i*4+1:i*4+5]
			neighbors = np.nan*np.ones(((num_agents-1)*4))
			curr_neighbor = 0 
			for j in range(num_agents):
				if i != j:
					state_j = data[t,j*4+1:j*4+5]
					dist = np.linalg.norm(state_i[0:2] - state_j[0:2])
					if dist <= neighborDist:
						neighbors[curr_neighbor*4:(curr_neighbor+1)*4] = state_i - state_j
						curr_neighbor += 1
			# desired control is the velocity in the next timestep
			u = data[t+1, i*4+3:i*4+5]

			X[t,:] = np.hstack((state_i,neighbors))
			Y[t,:] = u

	X = torch.tensor(X).float()
	Y = torch.tensor(Y).float()
	return X,Y


class DeepSetController(nn.Module):

	def __init__(self, state_dim, action_dim, hidden_layer):
		super(DeepSetController, self).__init__()
		
		self.phi = Phi(state_dim,hidden_layer)
		self.rho = Rho(state_dim,action_dim,hidden_layer)
		
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_layer = hidden_layer

	def forward(self,x):

		ndata = x.shape[0]
		X = torch.zeros((ndata,self.hidden_layer+self.state_dim))

		for i_x in range(ndata):
			x_i = x[i_x,:]
			s,neighbors = self.unpack(x_i)
			summ = torch.zeros((self.hidden_layer))
			for neighbor_i in range(neighbors.shape[1]):
				neighbor = neighbors[:,neighbor_i]
				summ += self.phi(neighbor)
			X[i_x,:] = torch.cat((s,summ))

		return self.rho(X)

	def unpack(self,x):
		s = x[0:self.state_dim]
		neighbors = x[self.state_dim:]
		neighbors = neighbors[~torch.isnan(neighbors)]
		n_neighbors = int(len(neighbors)/self.state_dim)
		neighbors = neighbors.reshape((self.state_dim,n_neighbors))
		return s,neighbors



class Phi(nn.Module):

	def __init__(self,state_dim,hidden_layer):
		super(Phi, self).__init__()

		self.fc1 = nn.Linear(state_dim, 32)
		self.fc2 = nn.Linear(32, hidden_layer)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x


class Rho(nn.Module):

	def __init__(self,state_dim,action_dim,hidden_layer):
		super(Rho, self).__init__()

		self.fc1 = nn.Linear(state_dim + hidden_layer, 32)
		self.fc2 = nn.Linear(32, action_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x



def train(param, model, loader):

	il_lr = 0.005
	optimizer = torch.optim.Adam(model.parameters(), lr=il_lr)
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


if __name__ == '__main__':

	torch.manual_seed(1)    # pytorch 
	
	state_dim = 4 
	action_dim = 2 
	hidden_layer = 32
	il_batch_size = 2000
	il_n_epoch = 500
	il_log_interval = 1
	il_train_model_fn = 'temp.pt'

	model = DeepSetController(state_dim,action_dim,hidden_layer)

	print('loading dataset...')
	x_train,y_train = load_dataset_v2("../baseline/orca/build/orca_ring10.npy")
	print('done')

	# x_train,y_train = make_dataset(param, env)
	dataset_train = Data.TensorDataset(x_train, y_train)
	loader_train = Data.DataLoader(
		dataset=dataset_train, 
		batch_size=il_batch_size, 
		shuffle=True)

	best_test_loss = np.Inf
	for epoch in range(1,il_n_epoch+1):
		train_epoch_loss = train([], model, loader_train)
		if epoch%il_log_interval==0:
			print('epoch: ', epoch)
			print('   Train Epoch Loss: ', train_epoch_loss)
			if train_epoch_loss < best_test_loss:
				best_test_loss = train_epoch_loss
				print('      saving @ best test loss:', best_test_loss)
				torch.save(model,il_train_model_fn)