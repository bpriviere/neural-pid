import subprocess
import numpy as np

if __name__ == '__main__':
	for i in range(50):
		print(i)
		# run process
		subprocess.run('./orca')
		# load file and convert to binary
		data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)
		# store in binary format
		with open("orca{}.npy".format(i), "wb") as f:
			np.save(f, data, allow_pickle=False)