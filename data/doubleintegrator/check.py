import glob
import os
import shutil
import stats
import numpy as np
import yaml

if __name__ == '__main__':

	if not os.path.exists("central3_invalid"):
		os.mkdir("central3_invalid")

	for file in sorted(glob.glob("central3/*.npy", recursive=True)):
		instance = os.path.splitext(os.path.basename(file))[0]
		map_filename = "../singleintegrator/instances3/{}.yaml".format(instance)
		result = stats.stats(map_filename, file)

		print(instance)

		if result["percent_agents_reached_goal"] < 100 or result["num_collisions"] > 0:
			print("Failed!", instance)
			shutil.move(file, "central3_invalid/{}.npy".format(instance))
