import glob
import os
import numpy as np
import yaml
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
import matplotlib.animation as manimation
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


class Animation:
	def __init__(self, instance, schedule, duration):

		with open(instance) as map_file:
			map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

		aspect = (map_data["map"]["dimensions"][0] + 2) / (map_data["map"]["dimensions"][1] + 2)
		self.fig, self.ax = plt.subplots(frameon=False, figsize=(4 * aspect, 4))
		self.ax.set_aspect('equal')
		plt.axis('off')
		self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=None, hspace=None)

		xmin = -1
		ymin = -1
		xmax = map_data["map"]["dimensions"][0] + 1
		ymax = map_data["map"]["dimensions"][1] + 1

		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)

		for o in map_data["map"]["obstacles"]:
			self.ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
		for x in range(-1,map_data["map"]["dimensions"][0]+1):
			self.ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
			self.ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
		for y in range(map_data["map"]["dimensions"][1]):
			self.ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
			self.ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

		self.data = np.load(schedule)
		self.num_agents = len(map_data["agents"])
		self.dt = self.data[1,0] - self.data[0,0]
		self.colors = []
		self.robots = []
		for i in range(self.num_agents):
			# plot trajectory
			# line = self.ax.plot(self.data[:,1+i*4], self.data[:,1+i*4+1],alpha=0.5,linewidth=2,linestyle='dashed')
			line = self.ax.plot(self.data[0:0,1+i*4], self.data[0:0,1+i*4+1],alpha=0.5,linewidth=2,linestyle='dashed')
			color = line[0].get_color()
			self.colors.append(color)

			# plot start and goal
			start = np.array(map_data["agents"][i]["start"])
			goal = np.array(map_data["agents"][i]["goal"])
			self.robots.append(Circle(start + np.array([0.5,0.5]), 0.15, alpha=0.5, color=color))
			self.ax.add_patch(self.robots[-1])
			self.ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))

		if duration < 0:
			duration = int(self.data[-1,0])
		else:
			duration = int(duration)

		self.anim = animation.FuncAnimation(self.fig, self.animate_func,
															 init_func=self.init_func,
															 frames=duration * 10,
															 interval=100,
															 blit=True)


	def save(self, file_name, speed):
		self.anim.save(
			file_name,
			"ffmpeg",
			fps=10 * speed,
			dpi=300)
			# savefig_kwargs={"pad_inches": 0, "bbox_inches": 0})

	def show(self):
		plt.show()

	def init_func(self):
		return self.robots
		# for p in self.robots:
		# 	self.ax.add_patch(p)
		# for a in self.artists:
		# 	self.ax.add_artist(a)
		# return self.patches + self.artists

	def animate_func(self, i):
		idx = min(i * int(0.1 / self.dt), self.data.shape[0] - 1)
		for k in range(self.num_agents):
			self.robots[k].center = self.data[idx,1+4*k+0:1+4*k+2]

		return self.robots


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("instance", help="input file containing instance (yaml)")
	parser.add_argument("schedule", help="input file containing trajectories (npy)")
	parser.add_argument('--video', dest='video', default=None, help="output video file (or leave empty to show on screen)")
	parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
	parser.add_argument("--duration", type=float, default=-1, help="duration of video")
	args = parser.parse_args()

	import sys
	sys.path.insert(1, os.path.join(os.getcwd(),'../../results/singleintegrator'))
	import stats

	results = stats.stats(args.instance, args.schedule)
	print(args.schedule, results["percent_agents_success"])

	animation = Animation(args.instance, args.schedule, args.duration)

	if args.video:
		animation.save(args.video, args.speed)
	else:
		animation.show()
