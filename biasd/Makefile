all: biasd_gsl biasd_cuda biasd_adaptive forward_backward

biasd_adaptive: ./src/biasd_ll_adaptive.c ./src/biasd_adaptive.h
	mkdir -p lib
	gcc -shared -fPIC -O3 -o ./lib/biasd_ll_adaptive.so ./src/biasd_ll_adaptive.c -lm

biasd_cuda: ./src/biasd_ll_cuda.cu 
	mkdir -p lib
	nvcc --shared --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" ./src/biasd_ll_cuda.cu -o ./lib/biasd_ll_cuda.so 

biasd_gsl: ./src/biasd_ll_gsl.c
	mkdir -p lib
	gcc -shared -fPIC -O3 -I/usr/local/include/gsl ./src/biasd_ll_gsl.c -o ./lib/biasd_ll_gsl.so -lgsl -lgslcblas -lm

forward_backward: ./src/forward_backward.c
	mkdir -p lib
	gcc -shared -fPIC -O3 -o ./lib/forward_backward.so ./src/forward_backward.c -lm

clean:
	rm ./lib/biasd_ll_cuda.so ./lib/biasd_ll_gsl.so ./lib/biasd_ll_adaptive.so ./lib/forward_backward.so
