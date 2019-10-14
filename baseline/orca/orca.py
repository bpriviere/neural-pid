import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

  data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)

  print(data.dtype)

  # store in binary format
  with open("orca.npy", "wb") as f:
    np.save(f, data, allow_pickle=False)

  num_agents = int((data.shape[1] - 1) / 4)
  print(num_agents)

  fig, ax = plt.subplots()
  for i in range(num_agents):
    ax.plot(data[:,i*4+1], data[:,i*4+2])
  plt.show()
