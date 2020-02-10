all: gen nn-export-test swig

gen:
	./gen.sh

nn-export-test:
	gcc -std=c99 nn.c main.c -lm -o nn-export-test

swig:
	swig -python nnexport.i
	gcc -std=c99 -shared -fPIC \
		nnexport_wrap.c \
		nn_double_integrator.c \
		-I/home/ben/anaconda3/envs/neural_pid_env/include/python3.6m \
		-Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -O3 -ffunction-sections -pipe -fdebug-prefix-map=/home/conda/feedstock_root/build_artifacts/python_1562031814771/work=/usr/local/src/conda/python-3.6.7 -fdebug-prefix-map=/home/ben/anaconda3/envs/neural_pid_env=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall \
		-o _nnexport.so

clean:
	rm -f generated_weights.c nn-export-test nnexport.py nnexport.pyc _nnexport.so __init__.pyc
