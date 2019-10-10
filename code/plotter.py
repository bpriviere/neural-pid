
from param import param
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages


def show():
	plt.show()


def save_figs():	
	fn = os.path.join( os.getcwd(), param.plots_fn)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs():
	pdf_path = os.path.join( os.getcwd(), param.plots_fn)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])


def plot(T,X,title=None,fig=None,ax=None,label=None):
	
	if fig is None or ax is None:
		fig, ax = plt.subplots()
	
	if label is not None:
		ax.plot(T,X,label=label)
		ax.legend()
	else:
		ax.plot(T,X)
	
	if title is not None:
		ax.set_title(title)

	return fig, ax


def make_fig(axlim = None):
	fig, ax = plt.subplots()
	if axlim is None:
		ax.set_aspect('equal')
	else:
		ax.set_xlim(-axlim[0],axlim[0])
		ax.set_ylim(-axlim[1],axlim[1])
		ax.set_autoscalex_on(False)
		ax.set_autoscaley_on(False)
		ax.set_aspect('equal')
	return fig, ax



def plot_circle(x,y,r,fig=None,ax=None,title=None,label=None):
	if fig is None or ax is None:
		fig, ax = plt.subplots()

	if label is not None:
		circle = patches.Circle((x,y),radius=r,label=label)
		ax.add_artist(circle)

	else:
		circle = patches.Circle((x,y),radius=r)
		ax.add_artist(circle)

	if title is not None:
		ax.set_title(title)

	return fig,ax
	

def plot_ss(env,states):
	fig,ax = env.render()
	ax.plot(states[:,0],states[:,1],linestyle='dashed')



def visualize(env,states):
	if param.env_name is 'CartPole':
		visualize_cartpole(env,states)


def visualize_cartpole(env,states):

	import meshcat
	import meshcat.geometry as g
	import meshcat.transformations as tf
	import time

	times = param.sim_times

	# visualize 3D
	if visualize:
		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		vis["cart"].set_object(g.Box([0.2,0.5,0.2]))
		vis["pole"].set_object(g.Cylinder(env.length_pole, 0.01))

		while True:
			for t, state in zip(times, states):
				vis["cart"].set_transform(tf.translation_matrix([0, state[0], 0]))

				vis["pole"].set_transform(
					tf.translation_matrix([0, state[0] + env.length_pole/2, 0]).dot(
					tf.rotation_matrix(np.pi/2 + state[1], [1,0,0], [0,-env.length_pole/2,0])))

				time.sleep(0.1)