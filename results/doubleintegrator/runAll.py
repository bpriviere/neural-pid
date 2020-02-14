#!/usr/bin/env python3
import glob
import os
import subprocess
import numpy as np
import argparse
import concurrent.futures
import tempfile
from itertools import repeat

def rollout_instance(file, args):
  print(file)
  subprocess.run("python3 examples/run_doubleintegrator.py -i {} {} --batch".format(os.path.abspath(file), args),
    cwd="../../code",
    shell=True)    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--barrier", action='store_true')
  parser.add_argument("--empty", action='store_true')
  parser.add_argument("--apf", action='store_true')
  args = parser.parse_args()

  agents_lst = [4,16,64] #[2,4,8,16,32,64]
  obst_lst = [6] #[6,12]

  datadir = []
  for agents in agents_lst:
    for obst in obst_lst:
      datadir.extend(glob.glob("../singleintegrator/instances/*obst{}_agents{}_*".format(obst,agents)))

  datadir = sorted(datadir)

  # name, kind, path
  controllers = []
  if args.apf:
    controllers.append("apf,apf,none")
  if args.barrier:
    controllers.append("exp1Barrier_0,torch,../results/doubleintegrator/exp1Barrier_0/il_current.pt")
  if args.empty:
    controllers.append("exp1Empty_0,EmptyAPF,../results/doubleintegrator/exp1Empty_0/il_current.pt")

  rolloutargs = ""
  for controller in controllers:
    rolloutargs += " --controller {}".format(controller)

  # print(rolloutargs)
  # exit()

  # serial version
  # for file in datadir:
  # for file in ['instances/map_8by8_obst6_agents4_ex0006.yaml']:
    # rollout_instance(file,rolloutargs)
    # exit()

  # parallel version
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    for _ in executor.map(rollout_instance, datadir, repeat(rolloutargs)):
      pass
