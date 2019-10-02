
from param import param
import matplotlib.pyplot as plt
import numpy as np 
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages


def show():
	plt.show()

def save_figs():
	

	if param.plots_combine_on:
		fn = os.path.join( os.getcwd(), param.plots_fn+'.pdf')
		pp = PdfPages(fn)
		for i in plt.get_fignums():
			pp.savefig(plt.figure(i))
			plt.close(plt.figure(i))
		pp.close()

	else:
		for i in plt.get_fignums():
			fn = os.path.join( os.getcwd(), param.plots_fn+str(i)+'.pdf')
			pp = PdfPages(fn)
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
		plt.plot(T,X,label=label)
		plt.legend()
	else:
		plt.plot(T,X)
	
	if title is not None:
		plt.title(title)

	return fig, ax

def make_fig():
	fig, ax = plt.subplots()
	plt.axis('equal')
	return fig, ax