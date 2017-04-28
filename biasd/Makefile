all: biasd_gsl biasd_adaptive biasd_cuda

biasd_adaptive: ./src/biasd_ll_adaptive.c ./src/biasd_adaptive.h
	mkdir -p lib
	gcc -shared -fPIC -O3 -o ./lib/biasd_ll_adaptive.so ./src/biasd_ll_adaptive.c -lm

biasd_cuda: ./src/biasd_ll_cuda.cu
	mkdir -p lib
	nvcc --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o ./src/biasd_twostate_integrate_cuda.o -c ./src/biasd_twostate_integrate_cuda.cu
	nvcc --shared --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o ./lib/biasd_ll_cuda.so ./src/biasd_twostate_ll_cuda.cu ./src/biasd_twostate_integrate_cuda.o
	rm ./src/biasd_twostate_integrate_cuda.o

biasd_gsl: ./src/biasd_ll_gsl.c
	mkdir -p lib
	gcc -shared -fPIC -O3 -I/usr/local/include/gsl ./src/biasd_ll_gsl.c -o ./lib/biasd_ll_gsl.so -lgsl -lgslcblas -lm

clean:
	rm -rf ./lib
