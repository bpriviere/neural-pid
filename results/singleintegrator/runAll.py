#!/usr/bin/env python3
import glob
import os
import subprocess
import numpy as np
import argparse
import concurrent.futures


def rollout_instance(file):

    basename = os.path.splitext(os.path.basename(file))[0]
    print(basename)
    if args.central:
      # Centralized planner
      if not os.path.exists(os.path.abspath("central/"+basename+".npy")):
        subprocess.run("./run.sh {} {}".format(
          os.path.abspath(file),
          os.path.abspath("central/"+basename+".npy")),
          shell=True,
          cwd="../../baseline/centralized-planner")

    if args.orca:
      # ORCA
      subprocess.run("../../baseline/orca/build/orca -i {}".format(file), shell=True)
      # load file and convert to binary
      data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)
      # store in binary format
      with open("orca/{}.npy".format(basename), "wb") as f:
          np.save(f, data, allow_pickle=False)

    if args.il:
      subprocess.run("python3 examples/run_singleintegrator.py -i {} --batch".format(os.path.abspath(file)),
        cwd="../../code",
        shell=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--orca", action='store_true')
  parser.add_argument("--central", action='store_true')
  parser.add_argument("--il", action='store_true')
  args = parser.parse_args()


  datadir = sorted(glob.glob("instances/*agents4_ex000*"))
  # ORCA and Central cannot be run in parallel (they use temporary files)
  if args.orca or args.central:
    for file in datadir:
      rollout_instance(file)
  else:
    with concurrent.futures.ProcessPoolExecutor(max_workers = 5) as executor:
      for _ in executor.map(rollout_instance, datadir):
        pass
