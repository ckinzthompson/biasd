all: c_biasd check_cuda

c_biasd: ./c_biasd.c ./c_biasd.h
	@echo "Building C:";
	gcc -shared -fPIC -O3 -o ./c_biasd.so ./c_biasd.c -lm

cuda_biasd: ./cuda_biasd.cu
	@echo "Building CUDA:";
	nvcc --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o ./cuda_biasd_integrate.o -c ./cuda_biasd_integrate.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
	nvcc --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o ./cuda_help.o -c ./cuda_help.cu -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
	nvcc --shared --relocatable-device-code=true --compiler-options '-fPIC' -D_FORCE_INLINES -Xcompiler "-O3" -o ./cuda_biasd.so ./cuda_biasd.cu ./cuda_biasd_integrate.o ./cuda_help.o -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
	@rm ./*.o

check_cuda:
	@if command -v nvcc > /dev/null 2>&1; then \
		$(MAKE) cuda_biasd; \
	else \
		echo "CUDA not found, skipping CUDA build"; \
	fi

clean:
	rm *.so
