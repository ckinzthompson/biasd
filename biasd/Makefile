all: biasd_c biasd_gsl biasd_cuda 

biasd_c: ./src/biasd_ll.c ./src/biasd_ll.h
	mkdir -p lib
	gcc -shared -fPIC -O3 -o ./lib/biasd_ll.so ./src/biasd_ll.c -lm

biasd_cuda: ./src/biasd_ll_cuda.cu ./src/biasd_ll.h
	mkdir -p lib
	nvcc --shared --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" ./src/biasd_ll_cuda.cu -o ./lib/biasd_ll_cuda.so 

biasd_gsl: ./src/biasd_ll_gsl.c
	mkdir -p lib
	gcc -shared -fPIC -O3 -I/usr/local/include/gsl ./src/biasd_ll_gsl.c -o ./lib/biasd_ll_gsl.so -lgsl -lgslcblas -lm

clean:
	rm ./lib/biasd_ll.so ./lib/biasd_ll_cuda.so ./lib/biasd_ll_gsl.so
