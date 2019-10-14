from run import run
from systems.cartpole import CartPole
from run_cartpole import CartpoleParam
from planning.rrt import rrt
from planning.scp import scp
import numpy as np
import sys, os

if __name__ == '__main__':
	param = CartpoleParam()
	param.env_case = 'Any90' #'SmallAngle','Swing90','Swing180', 'Any90'

	env = CartPole(param)

	for i in range(1000):

		k = 0
		while True:
			param.scp_fn = '../models/CartPole/dataset/scp_{}.csv'.format(k)
			param.scp_pdf_fn = '../models/CartPole/dataset/scp_{}.pdf'.format(k)
			if not os.path.isfile(param.scp_fn):
				break
			k += 1
		print("Running on ", param.scp_fn)

		rrt(param, env)

		# # fake rrt
		# x0 = env.reset()
		# xf = param.ref_trajectory[:,-1]
		# x = np.linspace(x0, xf, 50)
		# u = np.zeros((50,1))
		# data = np.hstack([x, u])
		# np.savetxt(param.rrt_fn, data, delimiter=',')

		scp(param, env)