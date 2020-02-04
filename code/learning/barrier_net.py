# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

# my package
from learning.deepset import DeepSet
# from learning.empty_net import Empty_Net
from learning.feedforward import FeedForward
from utilities import torch_tile, min_dist_circle_rectangle, torch_min_point_circle_rectangle, min_point_circle_rectangle


# class Barrier_Funcs():
# 	def __init__(self,param):
# 		self.param = param 
# 		self.device = torch.device('cpu')

# 		self.dim_neighbor = param.il_phi_network_architecture[0].in_features
# 		self.dim_action = param.il_psi_network_architecture[-1].out_features
# 		self.dim_state = param.il_psi_network_architecture[0].in_features - \
# 						param.il_rho_network_architecture[-1].out_features - \
# 						param.il_rho_obs_network_architecture[-1].out_features

# 		self.device = torch.device('cpu')

# 	def to(self,device):
# 		self.device = device

# 	def torch_fdbk_si(self,x,P,H):
# 		# todo 
# 		pass


# 	def numpy_fdbk_si(self,x,P,H):

# 		f = np.zeros((2,1))
# 		g = np.eye(2)

# 		grad_phi = self.numpy_get_grad_phi(x,P,H) # in 1x2
# 		phi = self.numpy_get_phi(x,P,H)

# 		Lf = np.zeros((1))

# 		Lg = np.zeros((1,2))
# 		Lg = np.dot(grad_phi,g)
# 		Lg_pinv = np.zeros((2,1))
# 		Lg_pinv = self.numpy_pinv_vec(Lg)

# 		K = self.param.b_gamma 
# 		eta = np.zeros((1,1))
# 		eta[0] = phi

# 		# fdbk linearization
# 		# b = np.dot(Lg_pinv, -Lf - np.dot(K,eta)).T - 1.0/self.param.b_eps * Lg

# 		b = -K*grad_phi - 1./self.param.b_eps*Lg
# 		return b 

# 	def numpy_pinv_vec(self,x):
# 		x_inv = x.T/np.linalg.norm(x)**2.
# 		# x_inv = x.T
# 		return x_inv

# 	def numpy_fdbk_di(self,x,P,H):

# 		# print('v',v)
# 		# exit()

# 		v = -1*x[0,3:5]

# 		f = np.zeros((4,1))
# 		f[0:2] = np.expand_dims(v,1) 

# 		g = np.zeros((4,2))
# 		g[2:4,:] = np.eye(2)

# 		grad_phi = self.numpy_get_grad_phi(x,P,H) # in 1x2
# 		gradp2_phi = self.numpy_get_gradp2_phi(x,P,H)
# 		phi = self.numpy_get_phi(x,P,H)
# 		phidot = np.dot(grad_phi, v)
		
# 		grad_phidot = np.zeros((1,4))
# 		grad_phidot[0,0:2] = np.dot(v,gradp2_phi)
# 		grad_phidot[0,2:4] = grad_phi

# 		Lf2 = np.zeros((1))
# 		Lf2 = np.dot(grad_phidot,f)

# 		LgLf = np.zeros((1,2))
# 		LgLf = np.dot(grad_phidot,g)
# 		LgLf_pinv = np.zeros((2,1))
# 		LgLf_pinv = self.numpy_pinv_vec(LgLf)

# 		K = np.array(self.param.b_k)
# 		eta = np.zeros((2,1))
# 		eta[0] = phi
# 		eta[1] = phidot


# 		# --------
# 		# fdbk linearization
# 		# b = np.dot(LgLf_pinv, -Lf2 - np.dot(K,eta)).T - 1/self.param.b_eps * LgLf
# 		# --------

# 		# --------
# 		# this is what the proof says
# 		q = (np.dot(v.T,np.dot(gradp2_phi,v) + phi))/np.dot(grad_phi,grad_phi.T)
# 		if np.dot(grad_phi,v) > 0:
# 			gamma = self.param.b_gamma 
# 			# gamma = np.max((self.param.b_gamma,q))
# 		elif np.dot(grad_phi,v) < 0:
# 			gamma = 0 
# 			# gamma = np.min((self.param.b_gamma,q))
# 		else:
# 			print(np.dot(grad_phi,v))
# 			gamma = 0 

# 		gamma = self.param.b_gamma
# 		# print('np.dot(grad_phi,v) < 0',np.dot(grad_phi,v) < 0)
# 		# gamma = q 
# 		b = -gamma*grad_phi -1/self.param.b_eps*np.dot(grad_phi,v)*grad_phi
# 		# --------

# 		# --------
# 		# this works 
# 		# b = -self.param.b_gamma * grad_phi - 1/self.param.b_eps * LgLf
# 		# --------


# 		return b 

# 	def torch_fdbk_di(self,x,P,H):
		
# 		bs = x.shape[0]

# 		v = -1*x[:,3:5]

# 		f = torch.zeros((bs,4,1),device=self.device)
# 		f[:,0:2,:] = v.unsqueeze(2)

# 		g = torch.zeros((bs,4,2),device=self.device)
# 		g[:,2:4,:] = torch.eye(2)

# 		grad_phi = self.torch_get_grad_phi(x,P,H)
# 		gradp2_phi = self.torch_get_gradp2_phi(x,P,H)
# 		phi = self.torch_get_phi(x,P,H)
# 		phidot = torch.bmm(grad_phi.unsqueeze(1),v.unsqueeze(2))

# 		grad_phidot = torch.zeros((bs,1,4),device=self.device)
# 		grad_phidot[:,0,0:2] = torch.bmm(v.unsqueeze(1),gradp2_phi).squeeze()
# 		grad_phidot[:,0,2:4] = grad_phi

# 		Lf2 = torch.zeros((bs,1),device=self.device)
# 		Lf2 = torch.bmm(grad_phidot,f)

# 		LgLf = torch.zeros((bs,1,2),device=self.device)
# 		LgLf = torch.bmm(grad_phidot,g)
# 		LgLf_pinv = torch.zeros((bs,2,1),device=self.device)
# 		LgLf_pinv = self.torch_pinv_vec(LgLf)

# 		K = torch.tensor(self.param.b_k,device=self.device)
# 		K = K.repeat(3,1).unsqueeze(1)
# 		eta = torch.zeros((bs,2,1),device=self.device)
# 		eta[:,0] = phi.unsqueeze(1)
# 		eta[:,1] = phidot.squeeze().unsqueeze(1)

# 		b = torch.bmm(LgLf_pinv, -Lf2 - torch.bmm(K,eta)).permute(0,2,1) - 1/self.param.b_eps * LgLf
# 		b = b.squeeze()

# 		return b

# 	def torch_pinv_vec(self,x):
# 		pinv_x = torch.mul( x.squeeze().unsqueeze(2),\
# 			torch.pow(torch.norm(x,dim=1,p=2).unsqueeze(2),-1))
# 		return pinv_x

# 	def torch_get_gradp2_phi(self,x,P,H):
# 		bs = x.shape[0] # batch size 
# 		gradp2_phi = torch.zeros((bs,2,2),device=self.device)
# 		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
# 			normP = torch.norm(P[:,j,:],p=2,dim=1).unsqueeze(1)
			
# 			f1 = torch.zeros((bs,1),device=self.device)
# 			f2 = torch.zeros((bs,1),device=self.device)
# 			f3 = torch.zeros((bs,2),device=self.device)
# 			grad_f1 = torch.zeros((bs,1,2),device=self.device)
# 			grad_f2 = torch.zeros((bs,1,2),device=self.device)
# 			grad_f3 = torch.zeros((bs,2,2),device=self.device)

# 			idx = (normP > 0).squeeze()
			
# 			f1[idx] = torch.pow(normP[idx],-1)
# 			f2[idx] = torch.pow(normP[idx]-self.param.r_agent,-1)
			
# 			if idx.nelement() == 1:
# 				idx = 0
# 			else:

# 				f3[idx] = P[idx,j,:]

# 				grad_f1[idx] = torch.mul(f3[idx],torch.pow(normP[idx],-3)).unsqueeze(1)
# 				grad_f2[idx] = torch.mul(f3[idx],torch.pow(torch.mul(normP[idx],torch.pow(f2[idx],2)),-1)).unsqueeze(1)
# 				grad_f3[idx] = -1*torch.eye(2)

# 				# if P.shape[0] != 1:
# 				f3 = f3.unsqueeze(2)

# 				gradp2_phi[idx] += \
# 					torch.mul(torch.mul(f1[idx],f2[idx]).unsqueeze(2),grad_f3[idx]) + \
# 					torch.mul(f1[idx].unsqueeze(2),torch.bmm(f3[idx],grad_f1[idx])) + \
# 					torch.mul(f1[idx].unsqueeze(2),torch.bmm(f3[idx],grad_f2[idx]))

# 		return gradp2_phi


# 	def numpy_get_gradp2_phi(self,x,P,H):
# 		gradp2_phi = np.zeros((2,2))
# 		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
# 			normp = np.linalg.norm(P[:,j,:])
# 			if normp > 0:
# 				f1 = 1/normp
# 				f2 = 1/(normp - self.param.r_agent)
# 				f3 = P[:,j,:].T # in 2x1 
# 				grad_f1 = f3.T / normp**3. # in 1x2
# 				grad_f2 = f3.T / (normp * f2**2)
# 				grad_f3 = -1*np.eye(2)
# 				gradp2_phi += f1*f2*grad_f3 + f1*np.dot(f3, grad_f1) + f1*np.dot(f3,grad_f2)
# 		return gradp2_phi

# 	# torch functions, optimzied for batch 
# 	def torch_get_phi(self,x,P,H):
# 		phi = torch.zeros((len(x)),device=self.device)
# 		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
# 			idx = H[:,j] > 0
# 			phi[idx] += -1*torch.log(H[idx,j])
# 		return phi 

# 	def torch_get_grad_phi_inv(self,x,P,H):
# 		grad_phi = self.torch_get_grad_phi(x,P,H)
# 		grad_phi_inv = torch.zeros(grad_phi.shape,device=self.device)
# 		idx = torch.norm(grad_phi,p=2,dim=1) != 0
# 		grad_phi_inv[idx] = \
# 			torch.mul(grad_phi[idx],\
# 			torch.pow(torch.norm(grad_phi[idx],p=2,dim=1),-2).unsqueeze(1))
# 		return grad_phi_inv

# 	def torch_get_grad_phi(self,x,P,H):
# 		grad_phi = torch.zeros((len(x),self.dim_action),device=self.device)
# 		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
# 			grad_phi += self.torch_get_grad_phi_contribution(P[:,j,:],H[:,j])
# 		return grad_phi

# 	def torch_get_grad_phi_contribution(self,P,H):
# 		grad_phi = torch.zeros(P.shape,device=self.device)
# 		normP = torch.norm(P,p=2,dim=1)
# 		normP = normP.unsqueeze(1)
# 		normP = torch_tile(normP,1,P.shape[1])
# 		H = H.unsqueeze(1)
# 		H = torch_tile(H,1,P.shape[1])
# 		idx = normP > 0 
# 		grad_phi[idx] = torch.mul(torch.mul(torch.pow(H[idx],-1),torch.pow(normP[idx],-1)),P[idx])\
# 			/(self.param.r_comm - self.param.r_agent)
# 		return grad_phi

# 	def torch_get_adaptive_scaling(self,x,empty_action,barrier_action,P,H):
# 		adaptive_scaling = torch.ones(H.shape[0],device=self.device)
# 		# print('H',H)
# 		if not H.nelement() == 0:
# 			minH = torch.min(H,dim=1)[0]
# 			normb = torch.norm(barrier_action,p=2,dim=1)
# 			normpi = torch.norm(empty_action,p=2,dim=1)
# 			idx = minH < self.param.Delta_R
# 			adaptive_scaling[idx] = torch.min(\
# 				torch.mul(normb[idx],torch.pow(normpi[idx],-1)),torch.ones(1,device=self.device))
# 		return adaptive_scaling.unsqueeze(1)

# 	def torch_scale(self,action,max_action):
# 		inv_alpha = action.norm(p=2,dim=1)/max_action
# 		inv_alpha = torch.clamp(inv_alpha,min=1)
# 		inv_alpha = inv_alpha.unsqueeze(0).T
# 		inv_alpha = torch_tile(inv_alpha,1,2)
# 		action = action*inv_alpha.pow_(-1)
# 		return action

# 	def torch_get_alpha_fdbk(self):
# 		phi_max = -1*np.log(self.param.Delta_R/(self.param.r_obs_sense-self.param.r_agent))
# 		alpha = np.min((phi_max*self.param.b_gamma,self.param.alpha_fdbk))
# 		return alpha

# 	def torch_get_relative_positions_and_safety_functions(self,x):

# 		nd = x.shape[0] # number of data points in batch 
# 		nn = self.get_num_neighbors(x)
# 		no = self.get_num_obstacles(x)

# 		P = torch.zeros((nd,nn+no,2),device=self.device) # pj - pi 
# 		H = torch.zeros((nd,nn+no),device=self.device) 
# 		curr_idx = 0

# 		for j in range(nn):
# 			idx = self.get_agent_idx_j(x,j)
# 			P[:,curr_idx,:] = x[:,idx] * (1 - self.param.r_agent * torch.pow(torch.norm(x[:,idx], p=2, dim=1).unsqueeze(1), -1))
# 			H[:,curr_idx] = (torch.norm(P[:,curr_idx,:], p=2, dim=1) - self.param.r_agent)/(self.param.r_comm - self.param.r_agent)
# 			curr_idx += 1 

# 		for j in range(no):
# 			idx = self.get_obstacle_idx_j(x,j)
# 			closest_point = torch_min_point_circle_rectangle(
# 				torch.zeros(2,device=self.device), 
# 				self.param.r_agent,
# 				x[:,idx] - torch.tensor([0.5,0.5],device=self.device), 
# 				x[:,idx] + torch.tensor([0.5,0.5],device=self.device))
# 			P[:,curr_idx,:] = closest_point
# 			H[:,curr_idx] = (torch.norm(closest_point, p=2, dim=1) - self.param.r_agent)/(self.param.r_comm - self.param.r_agent)
# 			curr_idx += 1

# 		return P,H 

# 	# numpy function, otpimized for rollout
# 	def numpy_get_phi(self,x,P,H):
# 		phi = np.zeros(1)
# 		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
# 			if H[:,j] > 0:
# 				phi += -np.log(H[:,j])
# 		return phi 

# 	def numpy_get_grad_phi_inv(self,x,P,H):
# 		grad_phi = self.numpy_get_grad_phi(x,P,H)
# 		grad_phi_inv = np.zeros(grad_phi.shape)
# 		if not np.linalg.norm(grad_phi) == 0:
# 			grad_phi_inv = grad_phi / np.linalg.norm(grad_phi)**2.
# 		return grad_phi_inv

# 	def numpy_get_grad_phi(self,x,P,H):
# 		grad_phi = np.zeros((len(x),self.dim_action))
# 		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
# 			grad_phi += self.numpy_get_grad_phi_contribution(P[:,j,:],H[:,j])
# 		return grad_phi

# 	def numpy_get_grad_phi_contribution(self,P,H):
# 		normp = np.linalg.norm(P)
# 		grad_phi_ji = 0.
# 		if normp > 0:
# 			grad_phi_ji = P/(H*normp)/(self.param.r_comm - self.param.r_agent)
# 		return grad_phi_ji

# 	def numpy_get_alpha_fdbk(self):
# 		phi_max = -1*np.log(self.param.Delta_R/(self.param.r_obs_sense-self.param.r_agent))
# 		alpha = np.min((phi_max*self.param.b_gamma,self.param.alpha_fdbk))
# 		return alpha

# 	def numpy_get_adaptive_scaling(self,x,empty_action,barrier_action,P,H):
# 		adaptive_scaling = 1.0 
# 		if not H.size == 0 and np.min(H) < self.param.Delta_R:
# 			normb = np.linalg.norm(barrier_action)
# 			normpi = np.linalg.norm(empty_action)
# 			adaptive_scaling = np.min((normb/normpi,1))
# 		return adaptive_scaling

# 	def numpy_scale(self,action,max_action):
# 		alpha = max_action/np.linalg.norm(action)
# 		alpha = np.min((alpha,1))
# 		action = action*alpha
# 		return action		

# 	def numpy_get_relative_positions_and_safety_functions(self,x):
		
# 		nd = x.shape[0] # number of data points in batch 
# 		nn = self.get_num_neighbors(x)
# 		no = self.get_num_obstacles(x) 

# 		P = np.zeros((nd,nn+no,2)) # pj - pi 
# 		H = np.zeros((nd,nn+no)) 
# 		curr_idx = 0

# 		for j in range(nn):
# 			idx = self.get_agent_idx_j(x,j)
# 			P[:,curr_idx,:] = x[:,idx] * (1 - self.param.r_agent / np.linalg.norm(x[:,idx]))
# 			H[:,curr_idx] = (np.linalg.norm(P[:,curr_idx,:]) - self.param.r_agent)/(self.param.r_obs_sense-self.param.r_agent)
# 			curr_idx += 1 

# 		for j in range(no):
# 			idx = self.get_obstacle_idx_j(x,j)
# 			closest_point = min_point_circle_rectangle(
# 				np.zeros(2), 
# 				self.param.r_agent,
# 				x[:,idx] - np.array([0.5,0.5]), 
# 				x[:,idx] + np.array([0.5,0.5]))
# 			P[:,curr_idx,:] = closest_point
# 			H[:,curr_idx] = (np.linalg.norm(closest_point) - self.param.r_agent)/(self.param.r_obs_sense-self.param.r_agent)
# 			curr_idx += 1
# 		return P,H 


# 	# helper fnc		
# 	def get_num_neighbors(self,x):
# 		return int(x[0,0])

# 	def get_num_obstacles(self,x):
# 		nn = self.get_num_neighbors(x)
# 		return int((x.shape[1] - 1 - self.dim_state - nn*self.dim_neighbor) / 2)  # number of obstacles 

# 	def get_agent_idx_j(self,x,j):
# 		idx = 1+self.dim_state + self.dim_neighbor*j+np.arange(0,2,dtype=int)
# 		return idx

# 	def get_obstacle_idx_j(self,x,j):
# 		nn = self.get_num_neighbors(x)
# 		idx = 1 + self.dim_state + self.dim_neighbor*nn+j*2+np.arange(0,2,dtype=int)
# 		return idx

# 	def get_agent_idx_all(self,x):
# 		nn = self.get_num_neighbors(x)
# 		idx = np.arange(1+self.dim_state,1+self.dim_state+self.dim_neighbor*nn,dtype=int)
# 		return idx

# 	def get_obstacle_idx_all(self,x):
# 		nn = self.get_num_neighbors(x)
# 		idx = np.arange((1+self.dim_state)+self.dim_neighbor*nn, x.size()[1],dtype=int)
# 		return idx

# 	def get_goal_idx(self,x):
# 		idx = np.arange(1,1+self.dim_state,dtype=int)
# 		return idx 
	



class Barrier_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Barrier_Net, self).__init__()
		self.model_neighbors = DeepSet(
			param.il_phi_network_architecture,
			param.il_rho_network_architecture,
			param.il_network_activation,
			param.env_name
			)
		self.model_obstacles = DeepSet(
			param.il_phi_obs_network_architecture,
			param.il_rho_obs_network_architecture,
			param.il_network_activation,
			param.env_name
			)
		self.psi = FeedForward(
			param.il_psi_network_architecture,
			param.il_network_activation)		

		self.param = param 
		
		# temp fix 
		# self.action_dim_per_agent = param.il_psi_network_architecture[-1].out_features
		# self.state_dim_per_agent = param.il_phi_network_architecture[0].in_features

		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation
		self.device = torch.device('cpu')

		self.dim_neighbor = param.il_phi_network_architecture[0].in_features
		self.dim_action = param.il_psi_network_architecture[-1].out_features
		self.dim_state = param.il_psi_network_architecture[0].in_features - \
						param.il_rho_network_architecture[-1].out_features - \
						param.il_rho_obs_network_architecture[-1].out_features


	def to(self, device):
		self.device = device
		self.model_neighbors.to(device)
		self.model_obstacles.to(device)
		return super().to(device)

	def policy(self,x):

		if True:
			grouping = dict()
			for i,x_i in enumerate(x):
				key = (int(x_i[0][0]), x_i.shape[1])
				if key in grouping:
					grouping[key].append(i)
				else:
					grouping[key] = [i]

			if len(grouping) < len(x):
				A = np.empty((len(x),self.dim_action))
				for key, idxs in grouping.items():
					batch = torch.Tensor([x[idx][0] for idx in idxs])
					a = self(batch)
					a = a.detach().numpy()
					for i, idx in enumerate(idxs):
						A[idx,:] = a[i]

				return A
			else:
				A = np.empty((len(x),self.dim_action))
				for i,x_i in enumerate(x):
					a_i = self(x_i)
					A[i,:] = a_i 
				return A

		else:
			A = np.empty((len(x),self.dim_action))
			for i,x_i in enumerate(x):
				a_i = self(x_i)
				A[i,:] = a_i 
			return A



	def __call__(self,x):

		if type(x) == torch.Tensor:

			if self.param.safety == "potential":
				P,H = self.torch_get_relative_positions_and_safety_functions(x)
				barrier_action = -1*self.param.b_gamma*self.torch_get_grad_phi(x,P,H)
				empty_action = self.empty(x)
				empty_action = self.torch_scale(empty_action, self.param.pi_max)
				adaptive_scaling = self.torch_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				action = torch.mul(adaptive_scaling,empty_action)+barrier_action 
				action = self.torch_scale(action, self.param.a_max)

			elif self.param.safety == "fdbk_si":
				P,H = self.torch_get_relative_positions_and_safety_functions(x)
				Phi = self.torch_get_phi(x,P,H)
				GradPhiInv = self.torch_get_grad_phi_inv(x,P,H)
				barrier_action = -1*self.param.b_gamma*Phi.unsqueeze(1)*GradPhiInv
				empty_action = self.empty(x)
				empty_action = self.torch_scale(empty_action, self.param.pi_max)
				
				# alpha_fdbk = self.torch_get_alpha_fdbk() 
				# action = alpha_fdbk*empty_action + barrier_action

				adaptive_scaling = self.torch_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				action = torch.mul(adaptive_scaling,empty_action)+barrier_action  
				action = self.torch_scale(action, self.param.a_max)

			elif self.param.safety == "fdbk_di":
				
				P,H = self.torch_get_relative_positions_and_safety_functions(x)
				barrier_action = self.torch_fdbk_di(x,P,H)
								
				empty_action = self.empty(x)
				empty_action = self.torch_scale(empty_action, self.param.pi_max)
				
				# adaptive_scaling = self.torch_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				# action = torch.mul(adaptive_scaling,empty_action)+barrier_action  
				
				action = empty_action + barrier_action
				action = self.torch_scale(action, self.param.a_max)				

			else:
				exit('self.param.safety: {} not recognized'.format(self.param.safety))				


		elif type(x) is np.ndarray:

			if self.param.safety == "potential":
				P,H = self.numpy_get_relative_positions_and_safety_functions(x)
				barrier_action = -1*self.param.b_gamma*self.numpy_get_grad_phi(x,P,H)
				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.numpy_scale(empty_action, self.param.pi_max)
				adaptive_scaling = self.numpy_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				action = adaptive_scaling*empty_action+barrier_action 
				action = self.numpy_scale(action, self.param.a_max)

			elif self.param.safety == "fdbk_si":
				P,H = self.numpy_get_relative_positions_and_safety_functions(x)
				
				Phi = self.numpy_get_phi(x,P,H)
				GradPhiInv = self.numpy_get_grad_phi_inv(x,P,H)
				barrier_action = -1*self.param.b_gamma*Phi*GradPhiInv

				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.numpy_scale(empty_action, self.param.pi_max)

				# alpha_fdbk = self.numpy_get_alpha_fdbk() 
				# alpha_fdbk = self.numpy_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				# action = alpha_fdbk*empty_action + barrier_action

				adaptive_scaling = self.numpy_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				action = adaptive_scaling*empty_action+barrier_action 
				action = self.numpy_scale(action, self.param.a_max)

			elif self.param.safety == "fdbk_di":

				P,H = self.numpy_get_relative_positions_and_safety_functions(x)
				barrier_action = self.numpy_fdbk_di(x,P,H)

				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.numpy_scale(empty_action, self.param.pi_max)

				adaptive_scaling = self.numpy_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				# action = adaptive_scaling*empty_action + barrier_action 
				action = empty_action + barrier_action 
				# action = self.numpy_scale(action, self.param.a_max)				

			else:
				exit('self.param.safety: {} not recognized'.format(self.param.safety))

		else:
			exit('type(x) not recognized: ', type(x))

		return action 

	def empty(self,x):
		# batches are grouped by number of neighbors (i.e., each batch has data with the same number of neighbors)
		# x is a 2D tensor, where the columns are: relative_goal, relative_neighbors, ...

		num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
		num_obstacles = int((x.size()[1] - (1 + self.dim_state + self.dim_neighbor*num_neighbors))/2)

		rho_neighbors = self.model_neighbors.forward(x[:,self.get_agent_idx_all(x)])
		rho_obstacles = self.model_obstacles.forward(x[:,self.get_obstacle_idx_all(x)])
		g = x[:,self.get_goal_idx(x)]

		x = torch.cat((rho_neighbors, rho_obstacles, g),1)
		x = self.psi(x)
		return x		

	def torch_fdbk_si(self,x,P,H):
		# todo 
		pass


	def numpy_fdbk_si(self,x,P,H):

		f = np.zeros((2,1))
		g = np.eye(2)

		grad_phi = self.numpy_get_grad_phi(x,P,H) # in 1x2
		phi = self.numpy_get_phi(x,P,H)

		Lf = np.zeros((1))

		Lg = np.zeros((1,2))
		Lg = np.dot(grad_phi,g)
		Lg_pinv = np.zeros((2,1))
		Lg_pinv = self.numpy_pinv_vec(Lg)

		K = np.array(self.param.b_k)
		eta = np.zeros((1,1))
		eta[0] = phi

		b = np.dot(Lg_pinv, -Lf - np.dot(K,eta)).T - 1.0/self.param.b_eps * Lg
		return b 

	def numpy_pinv_vec(self,x):
		x_inv = x.T/np.linalg.norm(x)**2.
		# x_inv = x.T
		return x_inv

	def numpy_fdbk_di(self,x,P,H):

		# print('v',v)
		# exit()

		v = -1*x[0,3:5]

		f = np.zeros((4,1))
		f[0:2] = np.expand_dims(v,1) 

		g = np.zeros((4,2))
		g[2:4,:] = np.eye(2)

		grad_phi = self.numpy_get_grad_phi(x,P,H) # in 1x2
		gradp2_phi = self.numpy_get_gradp2_phi(x,P,H)
		phi = self.numpy_get_phi(x,P,H)
		phidot = np.dot(grad_phi, v)
		
		grad_phidot = np.zeros((1,4))
		grad_phidot[0,0:2] = np.dot(v,gradp2_phi)
		grad_phidot[0,2:4] = grad_phi

		Lf2 = np.zeros((1))
		Lf2 = np.dot(grad_phidot,f)

		LgLf = np.zeros((1,2))
		LgLf = np.dot(grad_phidot,g)
		LgLf_pinv = np.zeros((2,1))
		LgLf_pinv = self.numpy_pinv_vec(LgLf)

		K = np.array(self.param.b_k)
		eta = np.zeros((2,1))
		eta[0] = phi
		eta[1] = phidot

		b = np.dot(LgLf_pinv, -Lf2 - np.dot(K,eta)).T - 1/self.param.b_eps * LgLf
		return b 

	def torch_fdbk_di(self,x,P,H):
		
		bs = x.shape[0]

		v = -1*x[:,3:5]

		f = torch.zeros((bs,4,1),device=self.device)
		f[:,0:2,:] = v.unsqueeze(2)

		g = torch.zeros((bs,4,2),device=self.device)
		g[:,2:4,:] = torch.eye(2)

		grad_phi = self.torch_get_grad_phi(x,P,H)
		gradp2_phi = self.torch_get_gradp2_phi(x,P,H)
		phi = self.torch_get_phi(x,P,H)
		phidot = torch.bmm(grad_phi.unsqueeze(1),v.unsqueeze(2))

		grad_phidot = torch.zeros((bs,1,4),device=self.device)
		grad_phidot[:,0,0:2] = torch.bmm(v.unsqueeze(1),gradp2_phi).squeeze()
		grad_phidot[:,0,2:4] = grad_phi

		Lf2 = torch.zeros((bs,1),device=self.device)
		Lf2 = torch.bmm(grad_phidot,f)

		LgLf = torch.zeros((bs,1,2),device=self.device)
		LgLf = torch.bmm(grad_phidot,g)
		LgLf_pinv = torch.zeros((bs,2,1),device=self.device)
		LgLf_pinv = self.torch_pinv_vec(LgLf)

		K = torch.tensor(self.param.b_k,device=self.device)
		K = K.repeat(3,1).unsqueeze(1)
		eta = torch.zeros((bs,2,1),device=self.device)
		eta[:,0] = phi.unsqueeze(1)
		eta[:,1] = phidot.squeeze().unsqueeze(1)

		b = torch.bmm(LgLf_pinv, -Lf2 - torch.bmm(K,eta)).permute(0,2,1) - 1/self.param.b_eps * LgLf
		b = b.squeeze()

		return b

	def torch_pinv_vec(self,x):
		pinv_x = torch.mul( x.squeeze().unsqueeze(2),\
			torch.pow(torch.norm(x,dim=1,p=2).unsqueeze(2),-1))
		return pinv_x

	def torch_get_gradp2_phi(self,x,P,H):
		bs = x.shape[0]
		gradp2_phi = torch.zeros((bs,2,2),device=self.device)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			normP = torch.norm(P[:,j,:],p=2,dim=1).unsqueeze(1)
			
			f1 = torch.zeros((bs,1),device=self.device)
			f2 = torch.zeros((bs,1),device=self.device)
			f3 = torch.zeros((bs,2),device=self.device)
			grad_f1 = torch.zeros((bs,1,2),device=self.device)
			grad_f2 = torch.zeros((bs,1,2),device=self.device)
			grad_f3 = torch.zeros((bs,2,2),device=self.device)

			idx = (normP > 0).squeeze()
			
			f1[idx] = torch.pow(normP[idx],-1)
			f2[idx] = torch.pow(normP[idx]-self.param.r_agent,-1)
			
			if idx.nelement() == 1:
				idx = 0
			else:

				f3[idx] = P[idx,j,:]

				grad_f1[idx] = torch.mul(f3[idx],torch.pow(normP[idx],-3)).unsqueeze(1)
				grad_f2[idx] = torch.mul(f3[idx],torch.pow(torch.mul(normP[idx],torch.pow(f2[idx],2)),-1)).unsqueeze(1)
				grad_f3[idx] = -1*torch.eye(2)

				# if P.shape[0] != 1:
				f3 = f3.unsqueeze(2)

				gradp2_phi[idx] += \
					torch.mul(torch.mul(f1[idx],f2[idx]).unsqueeze(2),grad_f3[idx]) + \
					torch.mul(f1[idx].unsqueeze(2),torch.bmm(f3[idx],grad_f1[idx])) + \
					torch.mul(f1[idx].unsqueeze(2),torch.bmm(f3[idx],grad_f2[idx]))

		return gradp2_phi

	def numpy_get_gradp2_phi(self,x,P,H):
		gradp2_phi = np.zeros((2,2))
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			normp = np.linalg.norm(P[:,j,:])
			if normp > 0:
				f1 = 1/normp
				f2 = 1/(normp - self.param.r_agent)
				f3 = P[:,j,:].T # in 2x1 
				grad_f1 = f3.T / normp**3. # in 1x2
				grad_f2 = f3.T / (normp * f2**2)
				grad_f3 = -1*np.eye(2)
				gradp2_phi += f1*f2*grad_f3 + f1*np.dot(f3, grad_f1) + f1*np.dot(f3,grad_f2)
		return gradp2_phi


	# torch functions, optimzied for batch 
	def torch_get_phi(self,x,P,H):
		phi = torch.zeros((len(x)),device=self.device)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			idx = H[:,j] > 0
			phi[idx] += -1*torch.log(H[idx,j])
		return phi 

	def torch_get_grad_phi_inv(self,x,P,H):
		grad_phi = self.torch_get_grad_phi(x,P,H)
		grad_phi_inv = torch.zeros(grad_phi.shape,device=self.device)
		idx = torch.norm(grad_phi,p=2,dim=1) != 0
		grad_phi_inv[idx] = \
			torch.mul(grad_phi[idx],\
			torch.pow(torch.norm(grad_phi[idx],p=2,dim=1),-2).unsqueeze(1))
		return grad_phi_inv

	def torch_get_grad_phi(self,x,P,H):
		grad_phi = torch.zeros((len(x),self.dim_action),device=self.device)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			grad_phi += self.torch_get_grad_phi_contribution(P[:,j,:],H[:,j])
		return grad_phi

	# def torch_get_grad_phi_contribution(self,P,H):
	# 	normP = torch.norm(P,p=2,dim=1)
	# 	normP = normP.unsqueeze(1)
	# 	normP = torch_tile(normP,1,P.shape[1])
	# 	H = H.unsqueeze(1)
	# 	H = torch_tile(H,1,P.shape[1])
	# 	grad_phi = torch.mul(torch.mul(torch.pow(H,-1),torch.pow(normP,-1)),P)
	# 	return grad_phi

	def torch_get_grad_phi_contribution(self,P,H):
		grad_phi = torch.zeros(P.shape,device=self.device)
		normP = torch.norm(P,p=2,dim=1)
		normP = normP.unsqueeze(1)
		normP = torch_tile(normP,1,P.shape[1])
		H = H.unsqueeze(1)
		H = torch_tile(H,1,P.shape[1])
		idx = normP > 0 
		grad_phi[idx] = torch.mul(torch.mul(torch.pow(H[idx],-1),torch.pow(normP[idx],-1)),P[idx])
		return grad_phi		

	def torch_get_adaptive_scaling(self,x,empty_action,barrier_action,P,H):
		adaptive_scaling = torch.ones(H.shape[0],device=self.device)
		# print('H',H)
		if not H.nelement() == 0:
			minH = torch.min(H,dim=1)[0]
			normb = torch.norm(barrier_action,p=2,dim=1)
			normpi = torch.norm(empty_action,p=2,dim=1)
			idx = minH < self.param.Delta_R
			adaptive_scaling[idx] = torch.min(\
				torch.mul(normb[idx],torch.pow(normpi[idx],-1)),torch.ones(1,device=self.device))
		return adaptive_scaling.unsqueeze(1)

	def torch_scale(self,action,max_action):
		inv_alpha = action.norm(p=2,dim=1)/max_action
		inv_alpha = torch.clamp(inv_alpha,min=1)
		inv_alpha = inv_alpha.unsqueeze(0).T
		inv_alpha = torch_tile(inv_alpha,1,2)
		action = action*inv_alpha.pow_(-1)
		return action

	def torch_get_alpha_fdbk(self):
		phi_max = -1*np.log(self.param.Delta_R/(self.param.r_obs_sense-self.param.r_agent))
		alpha = np.min((phi_max*self.param.b_gamma,self.param.alpha_fdbk))
		return alpha

	def torch_get_relative_positions_and_safety_functions(self,x):

		nd = x.shape[0] # number of data points in batch 
		nn = self.get_num_neighbors(x)
		no = self.get_num_obstacles(x)

		P = torch.zeros((nd,nn+no,2),device=self.device) # pj - pi 
		H = torch.zeros((nd,nn+no),device=self.device) 
		curr_idx = 0

		for j in range(nn):
			# j+1 to skip relative goal entries, +1 to skip number of neighbors column
			idx = self.get_agent_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx] * (1 - self.param.r_agent * torch.pow(torch.norm(x[:,idx], p=2, dim=1).unsqueeze(1), -1))
			H[:,curr_idx] = (torch.norm(P[:,curr_idx,:], p=2, dim=1) - self.param.r_agent)/(self.param.r_comm - self.param.r_agent)

			# H[:,curr_idx] = torch.max(torch.norm(x[:,idx], p=2, dim=1) - 2*self.param.r_agent, torch.zeros(1,device=self.device))
			# H[:,curr_idx] = torch.max(
			# 	(torch.norm(x[:,idx], p=2, dim=1) - 2*self.param.r_agent)/(self.param.r_comm-self.param.r_agent), 
			# 	self.param.eps_h*torch.ones(1,device=self.device))

			curr_idx += 1 

		for j in range(no):
			idx = self.get_obstacle_idx_j(x,j)
			closest_point = torch_min_point_circle_rectangle(
				torch.zeros(2,device=self.device), 
				self.param.r_agent,
				x[:,idx] - torch.tensor([0.5,0.5],device=self.device), 
				x[:,idx] + torch.tensor([0.5,0.5],device=self.device))
			P[:,curr_idx,:] = closest_point
			H[:,curr_idx] = (torch.norm(closest_point, p=2, dim=1) - self.param.r_agent)/(self.param.r_comm - self.param.r_agent)

			# H[:,curr_idx] = torch.max(torch.norm(closest_point, p=2, dim=1) - self.param.r_agent, torch.zeros(1,device=self.device))
			# H[:,curr_idx] = torch.max(
			# 	(torch.norm(closest_point, p=2, dim=1) - self.param.r_agent)/(self.param.r_comm-self.param.r_agent), 
			# 	self.param.eps_h*torch.ones(1,device=self.device))
			curr_idx += 1

		return P,H 

	# numpy function, otpimized for rollout
	def numpy_get_phi(self,x,P,H):
		phi = np.zeros(1)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			if H[:,j] > 0:
				phi += -np.log(H[:,j])
		return phi 

	def numpy_get_grad_phi_inv(self,x,P,H):
		grad_phi = self.numpy_get_grad_phi(x,P,H)
		grad_phi_inv = np.zeros(grad_phi.shape)
		if not np.linalg.norm(grad_phi) == 0:
			grad_phi_inv = grad_phi / np.linalg.norm(grad_phi)**2.
		return grad_phi_inv

	def numpy_get_grad_phi(self,x,P,H):
		grad_phi = np.zeros((len(x),self.dim_action))
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			grad_phi += self.numpy_get_grad_phi_contribution(P[:,j,:],H[:,j])
		return grad_phi

	def numpy_get_grad_phi_contribution(self,P,H):
		normp = np.linalg.norm(P)
		grad_phi_ji = 0.
		if normp > 0:
			grad_phi_ji = 1./(H*normp)*P
		return grad_phi_ji

	def numpy_get_alpha_fdbk(self):
		phi_max = -1*np.log(self.param.Delta_R/(self.param.r_obs_sense-self.param.r_agent))
		alpha = np.min((phi_max*self.param.b_gamma,self.param.alpha_fdbk))
		return alpha

	def numpy_get_adaptive_scaling(self,x,empty_action,barrier_action,P,H):
		adaptive_scaling = 1.0 
		if not H.size == 0 and np.min(H) < self.param.Delta_R:
			normb = np.linalg.norm(barrier_action)
			normpi = np.linalg.norm(empty_action)
			adaptive_scaling = np.min((normb/normpi,1))
		return adaptive_scaling

	def numpy_scale(self,action,max_action):
		alpha = max_action/np.linalg.norm(action)
		alpha = np.min((alpha,1))
		action = action*alpha
		return action		

	def numpy_get_relative_positions_and_safety_functions(self,x):
		
		nd = x.shape[0] # number of data points in batch 
		nn = self.get_num_neighbors(x)
		no = self.get_num_obstacles(x) 

		P = np.zeros((nd,nn+no,2)) # pj - pi 
		H = np.zeros((nd,nn+no)) 
		curr_idx = 0

		for j in range(nn):
			idx = self.get_agent_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx] * (1 - self.param.r_agent / np.linalg.norm(x[:,idx]))
			H[:,curr_idx] = (np.linalg.norm(P[:,curr_idx,:]) - self.param.r_agent)/(self.param.r_obs_sense-self.param.r_agent)
			curr_idx += 1 

		for j in range(no):
			idx = self.get_obstacle_idx_j(x,j)
			closest_point = min_point_circle_rectangle(
				np.zeros(2), 
				self.param.r_agent,
				x[:,idx] - np.array([0.5,0.5]), 
				x[:,idx] + np.array([0.5,0.5]))
			P[:,curr_idx,:] = closest_point
			H[:,curr_idx] = (np.linalg.norm(closest_point) - self.param.r_agent)/(self.param.r_obs_sense-self.param.r_agent)
			curr_idx += 1
		return P,H 


	# helper fnc		

	def get_num_neighbors(self,x):
		return int(x[0,0])

	def get_num_obstacles(self,x):
		nn = self.get_num_neighbors(x)
		return int((x.shape[1] - 1 - self.dim_state - nn*self.dim_neighbor) / 2)  # number of obstacles 

	def get_agent_idx_j(self,x,j):
		idx = 1+self.dim_state + self.dim_neighbor*j+np.arange(0,2,dtype=int)
		return idx

	def get_obstacle_idx_j(self,x,j):
		nn = self.get_num_neighbors(x)
		idx = 1 + self.dim_state + self.dim_neighbor*nn+j*2+np.arange(0,2,dtype=int)
		return idx

	def get_agent_idx_all(self,x):
		nn = self.get_num_neighbors(x)
		idx = np.arange(1+self.dim_state,1+self.dim_state+self.dim_neighbor*nn,dtype=int)
		return idx

	def get_obstacle_idx_all(self,x):
		nn = self.get_num_neighbors(x)
		idx = np.arange((1+self.dim_state)+self.dim_neighbor*nn, x.size()[1],dtype=int)
		return idx

	def get_goal_idx(self,x):
		idx = np.arange(1,1+self.dim_state,dtype=int)
		return idx 