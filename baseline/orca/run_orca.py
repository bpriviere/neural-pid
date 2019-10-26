import subprocess
import numpy as np

if __name__ == '__main__':
	for i in range(50):
		numAgents = int(np.random.uniform(1, 20))
		print(i, numAgents)
		# run process
		subprocess.run(['./orca', '--numAgents', str(numAgents), '--size', str(numAgents * 1.5)])
		# load file and convert to binary
		data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)
		# store in binary format
		with open("orca{}.npy".format(i), "wb") as f:
			np.save(f, data, allow_pickle=False)