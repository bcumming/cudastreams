NVCC_FLAGS=-arch=sm_35 --compiler-options -fopenmp
NVCC_FLAGS+=-DDEBUG_MSG
HEADERS=CudaStream.h CudaEvent.h

ifndef CRAY_CUDATOOLKIT_POST_LINK_OPTS
	LINKER_FLAGS=-lcuda -lcudart
else
	LINKER_FLAGS=$(CRAY_CUDATOOLKIT_POST_LINK_OPTS)
endif

all : transfer.exe

transfer.exe : transfer.cu $(HEADERS)
	nvcc -c transfer.cu $(NVCC_FLAGS)
	gcc transfer.o $(LINKER_FLAGS) -o transfer.exe -fopenmp

clean :
	rm -f *.exe
	rm -f *.o
