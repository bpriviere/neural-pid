import numpy as np
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os, shutil, glob
import datetime
import yaml
import itertools
from scp import scp
from robots import RobotCartpole


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  robot = RobotCartpole()

  data = np.loadtxt("rrt.csv", delimiter=',')

  goalState = [0,0,0,0]
  goalPos = [0,0]
  x, u = scp(robot, initialU = data[:,4:5], initialX = data[:,0:4], dt = 0.05, goalState=goalState, pdfFile = "scp_test.pdf")
  result = np.hstack([x, u])

  np.savetxt('scp.csv', result, delimiter=',')