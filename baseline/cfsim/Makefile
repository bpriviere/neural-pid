firmdir = /home/whoenig/projects/crazyflie/crazyflie-firmware
modinc = $(firmdir)/src/modules/interface
modsrc = $(firmdir)/src/modules/src

swig:
	swig -python -I$(modinc) -I$(firmdir)/src/hal/interface cffirmware.i
	gcc -std=c99 -shared -fPIC \
		-I$(modinc) \
		-I$(firmdir)/src/hal/interface \
		-I/usr/include/python3.6 \
		$(modsrc)/planner.c \
		$(modsrc)/pptraj.c \
		$(modsrc)/controller_sjc.c \
		cffirmware_wrap.c \
		-lm -lpython3.6m \
		-o _cffirmware.so

clean:
	rm -f cffirmware.py cffirmware.pyc _cffirmware.so __init__.pyc
