# execute from results folder!

import sys
import os
import argparse
sys.path.insert(1, os.path.join(os.getcwd(),'../code'))
sys.path.insert(1, os.path.join(os.getcwd(),'../code/examples'))
import run_doubleintegrator
from systems.doubleintegrator import DoubleIntegrator
from train_il import train_il
from other_policy import Empty_Net_wAPF, GoToGoalPolicy
from sim import run_sim
import torch
import concurrent.futures
from itertools import repeat
import glob
from multiprocessing import cpu_count
from torch.multiprocessing import Pool
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.getcwd(),'doubleintegrator'))
from createPlots import add_line_plot_agg, add_bar_agg, add_scatter
import stats
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--apf', action='store_true')
	args = parser.parse_args()

	agents_lst = [2,4,8,16,32,64]
	obst_lst = [6,12]

	if args.apf:

		folder = "doubleintegrator/apf"
		if not os.path.exists(folder):
			os.mkdir(folder)

		param = run_doubleintegrator.DoubleIntegratorParam()
		env = DoubleIntegrator(param)

		controller = {
			'apf': Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env))}

		for agent in agents_lst:
			for obst in obst_lst:
				files = glob.glob("singleintegrator/instances/*obst{}_agents{}_*.yaml".format(obst,agent), recursive=True)
				with Pool(cpu_count()) as p:
					p.starmap(run_doubleintegrator.run_batch, zip(repeat(param), repeat(env), files, repeat(controller)))
