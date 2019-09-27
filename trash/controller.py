
from param import param
from numpy import array, dot, ones

class PID_controller:

	def __init__(self):
		self.error_prev = None
		self.integral = None
		

	def eval(self,x,t):

		if param.get('system').get('name') is 'CartPole':
			xd = array([0,0,0,0])
			error = xd - x

			derivative = 0
			if self.error_prev is not None:
				derivative = (error - self.error_prev)/param.get('dt')

			integral = 0
			if self.integral is not None:
				integral = self.integral + error*param.get('dt')

			pid = param.get('kp')*error \
				+ param.get('kv')*derivative \
				+ param.get('ki')*integral

			K = ones((1,4))
			u = dot(K,pid)
			
			self.integral = integral
			self.error_prev = error

		return u
