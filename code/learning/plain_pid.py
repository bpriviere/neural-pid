import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from numpy import squeeze, array,arange, linspace
import numpy as np 

class PlainPID:
	"""
	Simple PID controller with fixed gains
	"""
	def __init__(self, Kp, Kd):
		self.Kp = Kp
		self.Kd = Kd

	def policy(self, state):
		action = (self.Kp[0]*state[0] + self.Kp[1]*state[1] + \
			self.Kd[0]*state[2] + self.Kd[1]*state[3])
		return action