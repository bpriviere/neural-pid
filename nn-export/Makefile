nn-export-test:
	gcc -std=c99 nn.c main.c -lm -o nn-export-test

swig:
	swig -python cffirmware.i
	gcc -std=c99 -shared -fPIC -DCMOCK \
		-I/usr/include/python3.6 \
		cffirmware_wrap.c \
		arm_math.c \
		kalman_core.c \
		ekf.c \
		-lm -lpython3.6m \
		-g \
		-o _cffirmware.so

clean:
	rm -f cffirmware.py cffirmware.pyc _cffirmware.so __init__.pyc
