NVCC_FLAGS=-arch=sm_35 --compiler-options -fopenmp

ifndef CRAY_CUDATOOLKIT_POST_LINK_OPTS
	LINKER_FLAGS=-lcuda -lcudart
else
	LINKER_FLAGS=$(CRAY_CUDATOOLKIT_POST_LINK_OPTS)
endif

futures.exe : futures.cu
	nvcc -c futures.cu $(NVCC_FLAGS)
	gcc futures.o $(LINKER_FLAGS) -o futures.exe -fopenmp

clean :
	rm -f *.exe
	rm -f *.o
