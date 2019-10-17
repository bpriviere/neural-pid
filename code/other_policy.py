
import numpy as np 

class ZeroPolicy:
	def __init__(self,env):
		self.env = env
	def policy(self,state):
		return np.zeros((self.env.m))

class LCP_Policy:
	# linear consensus protocol
	def __init__(self,env):
		self.env = env
	def policy(self,observation):
		a = np.zeros((self.env.m))
		dt = self.env.times[self.env.time_step+1] - self.env.times[self.env.time_step]
		for agent in self.env.agents:
			# observation_i = {s^j - s^i} \forall j in N^i
			a[agent.i] = sum(observation[agent.i])*dt	
		return a

class WMSR_Policy:

	def __init__(self,env):
		self.env = env

	# weighted mean subsequence reduced 
	def policy(self,observation):
		f = param.n_malicious
		a = np.zeros((self.env.m)) 
		dt = self.env.times[self.env.time_step+1] - self.env.times[self.env.time_step] 
		for agent in self.env.agents: 

			x_i = agent.x 
			x_js = observation[agent.i] + x_i 
			x_js_sorted = np.sort( np.array(x_js)) 

			ng = sum(x_i<=x_js)
			nl = sum(x_i>x_js)
		
			R = []
			if ng >= f:
				# add f largest values to R 
				for i in range(f):
					R.append(x_js_sorted[-(i+1)])
			else: 
				# add all values larger than x_i
				for x_j in x_js: 
					if x_j > x_i:
						R.append(x_j)

			if nl <= f:
				# add f smallest values to R
				for i in range(f):
					R.append(x_js_sorted[i])
			else:
				# add all values smaller than x_i
				for x_j in x_js:
					if x_j < x_i:
						R.append(x_i)
			
			R = np.array(R)

			count = 1
			summ = x_i
			for x_j in x_js:
				if not x_j in R:
					count += 1
					summ += x_j 

			a[agent.i] = -summ/count
		return a

