#!/usr/bin/env
import yaml
import numpy as np
import argparse
import os

import uav_trajectory

# this script first finds the common stretchtime (identical to multi-robot-trajectory-planning/tools/scaleTrajectory.py)
# and then outputs a single csv file with the sampled result

# returns maximum velocity, acceleration
def findMaxDynamicLimits(traj):
  vmax = 0
  amax = 0
  for t in np.arange(0, traj.duration, 0.1):
    e = traj.eval(t)
    vmax = max(vmax, np.linalg.norm(e.vel))
    amax = max(amax, np.linalg.norm(e.acc))
  return vmax, amax

# returns upper bound stretchtime factor
def upperBound(traj, vmax, amax):
  stretchtime = 1.0
  while True:
    v,a = findMaxDynamicLimits(traj)
    if v <= vmax and a <= amax:
      # print(v,a)
      return stretchtime
    traj.stretchtime(2.0)
    stretchtime = stretchtime * 2.0

# returns lower bound stretchtime factor
def lowerBound(traj, vmax, amax):
  stretchtime = 1.0
  while True:
    v,a = findMaxDynamicLimits(traj)
    if v >= vmax and a >= amax:
      # print(v,a)
      return stretchtime
    traj.stretchtime(0.5)
    stretchtime = stretchtime * 0.5

def findStretchtime(file, vmax, amax):
  traj = uav_trajectory.Trajectory()
  traj.loadcsv(file)
  L = lowerBound(traj, vmax, amax)
  traj.loadcsv(file)
  U = upperBound(traj, vmax, amax)
  while True:
    # print("L ", L)
    # print("U ", U)
    if U - L < 0.1:
      return U
    middle = (L + U) / 2
    # print("try: ", middle)
    traj.loadcsv(file)
    traj.stretchtime(middle)
    v,a = findMaxDynamicLimits(traj)
    # print("v,a ", v, a)
    if v <= vmax and a <= amax:
      U = middle
    else:
      L = middle


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("folder", type=str, help="input folder containing csv files")
  parser.add_argument("typesFile", help="types file for agent types (yaml)")
  parser.add_argument("agentsFile", help="agents file with agents (yaml)")
  args = parser.parse_args()

  with open(args.typesFile) as file:
    types = yaml.load(file)

  with open(args.agentsFile) as file:
    agents = yaml.load(file)

  agentTypes = dict()
  for agentType in types["agentTypes"]:
    agentTypes[agentType["type"]] = agentType

  result = 0.0
  for agent in agents["agents"]:
    name = agent["name"]
    agentType = agentTypes[agent["type"]]
    vmax = agentType["v_max"]
    amax = agentType["a_max"]
    stretchtime = findStretchtime(os.path.join(args.folder, name + ".csv"), vmax, amax)
    print(name, stretchtime)
    result = max(result, stretchtime)

  print("common stretchtime: {}".format(result))

  # export file
  with open('central.csv', 'w') as file:
    file.write("t")
    i = 0
    for _ in agents["agents"]:
      file.write(",x{},y{},vx{},vy{}".format(i,i,i,i))
      i += 1
    file.write("\n")
    # find total duration
    T = 0
    trajs = []
    for agent in agents["agents"]:
      name = agent["name"]
      traj = uav_trajectory.Trajectory()
      traj.loadcsv(os.path.join(args.folder, name + ".csv"))
      traj.stretchtime(result)
      T = max(T, traj.duration)
      trajs.append(traj)
    # write sampled data
    for t in np.arange(0, T, 0.25):
      file.write(str(t))
      for traj in trajs:
        e = traj.eval(t)
        file.write(",{},{},{},{}".format(e.pos[0], e.pos[1], e.vel[0], e.vel[1]))
      file.write("\n")
