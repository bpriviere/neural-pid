import argparse
import tempfile
import os
import subprocess

from discretePreProcessing import discretePreProcessing
from discretePostProcessing import discretePostProcessing
from exportTrajectories import exportTrajectories

def run(input_fn, output_fn):
	with tempfile.TemporaryDirectory() as tmpdirname:
		print('created temporary directory', tmpdirname)

		# discrete planning
		discretePreProcessing(input_fn, os.path.join(tmpdirname, "input.yaml"))

		subprocess.run(["./multi-robot-trajectory-planning/build/libMultiRobotPlanning/ecbs",
			"-i", os.path.join(tmpdirname, "input.yaml"),
			"-o", os.path.join(tmpdirname, "discreteSchedule.yaml"),
			"-w", "1.3"], timeout=60)

		# postprocess output (split paths)
		discretePostProcessing(
			os.path.join(tmpdirname, "input.yaml"),
			os.path.join(tmpdirname, "discreteSchedule.yaml"),
			os.path.join(tmpdirname, "discreteSchedule.yaml"))

		# convert yaml map -> octomap
		subprocess.run(["./multi-robot-trajectory-planning/build/tools/map2octomap/map2octomap",
			"-m", input_fn,
			"-o", os.path.join(tmpdirname, "map.bt")])

		# convert octomap -> STL (visualization)
		# (skip, since we do batch processing here)

		# continuous planning
		cmd = "path_setup,smoothener_batch('{}','{}','{}','{}'),quit".format(
			os.path.join(tmpdirname, "map.bt"),
			os.path.join(tmpdirname, "discreteSchedule.yaml"),
			"../examples/ground/types.yaml",
			os.path.join(tmpdirname) + "/")
		subprocess.run(["matlab",
			"-nosplash",
			"-nodesktop",
			"-r", cmd],
			cwd="multi-robot-trajectory-planning/smoothener")

		exportTrajectories(
			tmpdirname,
			"multi-robot-trajectory-planning/examples/ground/types.yaml",
			input_fn,
			output_fn)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="input file (yaml)")
	parser.add_argument("output", help="output file (npy)")
	args = parser.parse_args()

	run(args.input, args.output)