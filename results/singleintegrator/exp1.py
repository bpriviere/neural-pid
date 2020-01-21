#!/usr/bin/env python3

# execute from results folder!

import sys
import os
import argparse
sys.path.insert(1, os.path.join(os.getcwd(),'../code'))
sys.path.insert(1, os.path.join(os.getcwd(),'../code/examples'))
import run_singleintegrator
from systems.singleintegrator import SingleIntegrator
from train_il import train_il
from other_policy import Empty_Net_wAPF
from sim import run_sim
import torch
import concurrent.futures
from itertools import repeat
import glob
from multiprocessing import cpu_count
from torch.multiprocessing import Pool

sys.path.insert(1, os.path.join(os.getcwd(),'singleintegrator'))
from createPlots import add_line_plot_agg, add_bar_agg, add_scatter
import stats
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
  parser.add_argument('--train', action='store_true', help='run training')
  parser.add_argument('--sim', action='store_true', help='run validation inference')
  parser.add_argument('--plot', action='store_true', help='create plots')
  args = parser.parse_args()

  torch.multiprocessing.set_start_method('spawn')

  if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  agents_lst = [2,4,8,16,32]
  obst_lst = [6,9,12]

  if args.plot:
    solvers = ['orca', 'exp1Empty', 'central', 'exp1Barrier']
    solver_names = ['ORCA', 'APF', 'Central', 'Barrier']

    for obst in obst_lst:
      files = []
      result_by_instance = dict()
      for solver in solvers:
        for agent in agents_lst:
          files.extend( glob.glob("singleintegrator/{}*/*obst{}_agents{}_*.npy".format(solver,obst,agent), recursive=True))
      for file in files:
        instance = os.path.splitext(os.path.basename(file))[0]
        map_filename = "singleintegrator/instances/{}.yaml".format(instance)
        result = stats.stats(map_filename, file)
        result["solver"] = os.path.basename(result["solver"])

        if instance in result_by_instance:
          result_by_instance[instance].append(result)
        else:
          result_by_instance[instance] = [result]

      # create plots
      pp = PdfPages("exp1_{}.pdf".format(obst))

      add_line_plot_agg(pp, result_by_instance, "num_agents_success", "# robots success")
      add_line_plot_agg(pp, result_by_instance, "control_effort_sum", "control effort")
      add_scatter(pp, result_by_instance, "num_collisions", "# collisions")

      pp.close()
    exit()

  datadir = []
  for agents in agents_lst:
    for obst in obst_lst:
      datadir.extend(glob.glob("singleintegrator/instances/*obst{}_agents{}_*".format(obst,agents)))
  instances = sorted(datadir)



  for i in range(10):
      # train policy
      param = run_singleintegrator.SingleIntegratorParam()
      env = SingleIntegrator(param)
      if args.train:
        for cc in ['Empty', 'Barrier']:
          param = run_singleintegrator.SingleIntegratorParam()
          param.il_controller_class = cc
          param.il_train_model_fn = 'singleintegrator/exp1{}_{}/il_current.pt'.format(cc,i)
          env = SingleIntegrator(param)
          train_il(param, env, device)

      elif args.sim:
        # evaluate policy
        controllers = {
          'exp1Empty_{}'.format(i): Empty_Net_wAPF(param,env,torch.load('singleintegrator/exp1Empty_{}/il_current.pt'.format(i))),
          'exp1Barrier_{}'.format(i) : torch.load('singleintegrator/exp1Barrier_{}/il_current.pt'.format(i))
        }

        # for instance in instances:
          # run_singleintegrator.run_batch(instance, controllers)

        with Pool(24) as p:
          p.starmap(run_singleintegrator.run_batch, zip(instances, repeat(controllers)))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        #   for _ in executor.map(run_singleintegrator.run_batch, instances, repeat(controllers)):
        #     pass

