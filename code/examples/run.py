# hack to be able to load modules in Python3
import sys, os
sys.path.insert(1, os.path.join(os.getcwd(),'.'))

import argparse

from train_rl import train_rl
from train_il import train_il
from sim import sim
from planning.rrt import rrt
from planning.scp import scp

def run(param, env, controllers):
	parser = argparse.ArgumentParser()
	parser.add_argument("--rl", action='store_true')
	parser.add_argument("--il", action='store_true')
	parser.add_argument("--rrt", action='store_true')
	parser.add_argument("--scp", action='store_true')
	parser.add_argument("--animate", action='store_true')
	args = parser.parse_args()

	if args.rl:
		train_rl(param, env)
	elif args.il:
		train_il(param, env)
	elif args.rrt:
		rrt(param, env)
	elif args.scp:
		scp(param, env)
	else:
		sim(param, env, controllers, args.animate)
