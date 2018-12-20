all: biasd_c biasd_cuda

biasd_c: ./src/biasd_c.c ./src/biasd_c.h
	mkdir -p lib
	gcc -shared -fPIC -O3 -o ./lib/biasd_c.so ./src/biasd_c.c -lm

biasd_cuda: ./src/biasd_cuda.cu
	mkdir -p lib
	nvcc --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -arch=compute_50 -code=sm_50 -o ./src/biasd_cuda_integrate.o -c ./src/biasd_cuda_integrate.cu
	nvcc --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o -arch=compute_50 -code=sm_50 ./src/cuda_help.o -c ./src/cuda_help.cu
	nvcc --shared --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o -arch=compute_50 -code=sm_50 ./lib/biasd_cuda.so ./src/biasd_cuda.cu ./src/biasd_cuda_integrate.o ./src/cuda_help.o
	rm ./src/*.o

clean:
	rm -rf ./lib
