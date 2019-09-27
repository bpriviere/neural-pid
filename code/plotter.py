
from param import param
import matplotlib.pyplot as plt
import numpy as np 
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages


def show():
	plt.show()

def save_figs():
	fn = os.path.join( os.getcwd(), param.get('plots_fn'))

	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()

def open_figs():
	pdf_path = os.path.join( os.getcwd(), param.get('plots_fn'))
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])

def plot(T,X,title = None,fig = None,ax = None):
	if fig is None or ax is None:
		fig, ax = plt.subplots()
	plt.plot(T,X)
	if title is not None:
		plt.title(title)
	#TEMP
	# plt.xlim(0,1)
	return fig, ax

def make_fig():
	fig, ax = plt.subplots()
	plt.axis('equal')
	return fig, ax