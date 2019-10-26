
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages


def show():
	plt.show()


def save_figs(filename):
	fn = os.path.join( os.getcwd(), filename)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs(filename):
	pdf_path = os.path.join( os.getcwd(), filename)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])

def subplots():
	return plt.subplots()


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
		# ax.set_aspect('equal')
		pass
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

