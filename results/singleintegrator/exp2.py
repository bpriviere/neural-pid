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
# from torch.multiprocessing import Pool
from multiprocessing import Pool
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.getcwd(),'singleintegrator'))
from createPlots import add_line_plot_agg, add_bar_agg, add_scatter
import stats
from matplotlib.backends.backend_pdf import PdfPages

def rollout_instance(file, args):
  subprocess.run("python3 examples/run_singleintegrator.py -i {} {} --batch".format(os.path.abspath(file), args),
    cwd="../code",
    shell=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
  parser.add_argument('--train', action='store_true', help='run training')
  parser.add_argument('--sim', action='store_true', help='run validation inference')
  parser.add_argument('--plot', action='store_true', help='create plots')
  args = parser.parse_args()

  # torch.multiprocessing.set_start_method('spawn')

  if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  agents_lst = [4, 8, 16]
  obst_lst = [6,12]
  radii = [1,2,3,4,6,8]
  # training_data = [30000, 300000, 3000000]
  training_data = [30000, 300000, 3000000, 30000000]

  if args.plot:
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.linewidth'] = 4

    # default fig size is [6.4, 4.8]
    fig, axs = plt.subplots(1, len(obst_lst), sharex='all', sharey='row', figsize = [6.4 * 1.4, 4.8 * 1.4 * 0.6], squeeze=False)

    pp = PdfPages("exp2_collisions.pdf")
    for column, obst in enumerate(obst_lst):
      result_by_instance = dict()
      for r in radii:
        for agent in agents_lst:
          # load Empty
          for td in training_data:
            files = glob.glob("singleintegrator/exp2BarrierR{}td{}_*/*obst{}_agents{}_*.npy".format(r,td,obst,agent), recursive=True)
       
            for file in files:
              instance = os.path.splitext(os.path.basename(file))[0]
              map_filename = "singleintegrator/instances/{}.yaml".format(instance)
              result = stats.stats(map_filename, file)
              if td == 30000:
                result["solver"] = "|D| = 30 k"
              elif td == 300000:
                result["solver"] = "|D| = 300 k"
              elif td == 3000000:
                result["solver"] = "|D| = 3 M"
              elif td == 30000000:
                result["solver"] = "|D| = 30 M"
              else:
                result["solver"] = "Empty{}".format(td)
              result["Rsense"] = r

              if instance in result_by_instance:
                result_by_instance[instance].append(result)
              else:
                result_by_instance[instance] = [result]

          # load ORCA
          files = glob.glob("singleintegrator/orcaR{}*/*obst{}_agents{}_*.npy".format(r,obst,agent), recursive=True)
     
          for file in files:
            instance = os.path.splitext(os.path.basename(file))[0]
            map_filename = "singleintegrator/instances/{}.yaml".format(instance)
            result = stats.stats(map_filename, file)
            result["solver"] = "ORCA"
            result["Rsense"] = r

            if instance in result_by_instance:
              result_by_instance[instance].append(result)
            else:
              result_by_instance[instance] = [result]

      # add_bar_agg(pp, result_by_instance, "num_agents_success", "# robots success")
      add_line_plot_agg(None, result_by_instance, "percent_agents_success", group_by="Rsense",
        ax=axs[0, column])
      # add_line_plot_agg(pp, result_by_instance, "control_effort_sum", "control effort")
      add_scatter(pp, result_by_instance, "num_collisions", "# collisions")

    pp.close()

    # create plots
    pp = PdfPages("exp2.pdf")

    for column in range(0, 2):
      axs[0,column].set_xlabel("$r_{sense}$ [m]")
    
    axs[0,0].set_ylabel("robot success [%]")
    axs[0,0].set_ylim([25,105])

    axs[0,0].set_title("10 % obstacles")
    axs[0,1].set_title("20 % obstacles")

    for column in range(0, 2):
        axs[0, column].minorticks_off()
        axs[0, column].grid(which='major')

    fig.tight_layout()
    axs[0,0].legend()
    pp.savefig(fig)
    plt.close(fig)
    pp.close()


    exit()

  datadir = []
  for agents in agents_lst:
    for obst in obst_lst:
      datadir.extend(glob.glob("singleintegrator/instances/*obst{}_agents{}_*".format(obst,agents)))
  instances = sorted(datadir)


  for i in range(3,5):
    # train policy
    for r in radii:
      for td in training_data:
        if args.train:
          for cc in ['Barrier']:
            param = run_singleintegrator.SingleIntegratorParam()
            param.r_comm = r
            param.r_obs_sense = r
            param.max_neighbors = 6
            param.max_obstacles = 6
            param.il_load_loader_on = False
            param.il_controller_class = cc
            param.datadict["obst"] = td

            param.il_train_model_fn = 'singleintegrator/exp2{}R{}td{}_{}/il_current.pt'.format(cc,r,td,i)
            env = SingleIntegrator(param)
            if not os.path.exists(param.il_train_model_fn):
              train_il(param, env, device)
            del env
            del param

        elif args.sim:
          controller = "exp2BarrierR{0}td{1}_{2},torch,../results/singleintegrator/exp2BarrierR{0}td{1}_{2}/il_current.pt".format(r,td,i)
          rollout_args = "--controller {} --Rsense {} --maxNeighbors {}".format(controller, r, 6)

          with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            for _ in executor.map(rollout_instance, instances, repeat(rollout_args)):
              pass


          # torch.set_num_threads(1)

          # for instance in instances:
          #   run_singleintegrator.run_batch(param, env, instance, controllers)
          # with Pool(3) as p:
          #   p.starmap(run_singleintegrator.run_batch, zip(repeat(param), repeat(env), instances, repeat(controllers)))

          # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
          #   for _ in executor.map(run_singleintegrator.run_batch, repeat(param), repeat(env), instances, repeat(controllers)):
          #     pass

