LINKER_FLAGS=-lcuda -lcudart
NVCC_FLAGS=-arch=sm_35 --compiler-options -fopenmp

futures.exe : futures.cu
	nvcc futures.cu $(NVCC_FLAGS) $(LINKER_FLAGS) -o futures.exe

clean :
	rm -f *.exe
