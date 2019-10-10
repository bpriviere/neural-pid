from param import Param
from run import run
from systems.cartpole import CartPole

class CartpoleParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'CartPole'
		self.env_case = 'SmallAngle' #'SmallAngle','Swing90','Swing180'

		# RL
		self.rl_train_model_fn = 'rl_model.pt'

		# IL
		self.il_train_model_fn = 'il_model.pt'
		self.il_imitate_model_fn = 'rl_model.pt'
		self.kp = [2,40]
		self.kd = [2,40]

		# Sim
		self.sim_rl_model_fn = 'rl_model.pt'
		self.sim_il_model_fn = 'il_model.pt'


if __name__ == '__main__':
	param = CartpoleParam()
	env = CartPole(param)
	run(param, env)
