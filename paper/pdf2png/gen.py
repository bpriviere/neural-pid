import subprocess
import glob
import os

for file in glob.glob("*.pdf"):
	print(file)
	subprocess.run(["pdfcrop", file, file])
	subprocess.run(["pdftocairo", file, os.path.splitext(file)[0], "-png", "-singlefile", "-transp"])
