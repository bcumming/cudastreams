#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "CudaEvent.h"
#include "CudaStream.h"

// a = b+c
__global__
void sum(double* a, double* b, double* c, size_t N) {
    size_t tid = threadIdx.x + blockDim.x*blockIdx.x;
    size_t grid_step = blockDim.x * gridDim.x;

    while(tid<N) {
        a[tid] = b[tid] + c[tid];
        tid += grid_step;
    }
}

bool initialize_cuda() {
    CUresult result = cuInit(0);
    return result == CUDA_SUCCESS;
}

int main(void) {
    if(!initialize_cuda()) {
        std::cerr << "unable to initialize CUDA" << std::endl;
        return 1;
    }
    else {
        std::cout << "initialized CUDA" << std::endl;
    }

    CudaStream s(true);
    CudaStream sdefault(false);
    std::cout << "stream is " << (s.is_default_stream() ? "" : "not") << " the default stream : " << std::endl;

    {
        const size_t N=128*1024*1024;
        unsigned int block_dim = 128;
        unsigned int grid_dim = N/block_dim + (N%block_dim ? 1 : 0);
        grid_dim = grid_dim > 1024 ? 1024 : grid_dim;
        dim3 block(block_dim);
        dim3 grid(grid_dim);
        std::cout << "launch grid : " << block_dim << "*" << grid_dim << std::endl;

        double *a_d, *b_d, *c_d;
        cudaMalloc(&a_d, N*sizeof(double));
        cudaMalloc(&b_d, N*sizeof(double));
        cudaMalloc(&c_d, N*sizeof(double));

        double *a_h = new double[N];
        double *b_h = new double[N];
        double *c_h = new double[N];

        for(size_t i=0; i<N; ++i) {
            a_h[i] = 0.;
            b_h[i] = 1.;
            c_h[i] = 1.;
        }
        cudaMemcpy(a_d, a_h, N*sizeof(double),  cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b_h, N*sizeof(double),  cudaMemcpyHostToDevice);
        cudaMemcpy(c_d, c_h, N*sizeof(double),  cudaMemcpyHostToDevice);

        CudaEvent e;

        double time_init = omp_get_wtime();

        sum<<<block, grid, 0, s.stream()>>>(a_d, b_d, c_d, N);

        s.insert_event(e);

        double time_before_wait = omp_get_wtime();
        s.wait_on_event(e);
        double time_after_wait  = omp_get_wtime();
        e.wait();
        std::cout << "took " << time_before_wait-time_init << " " << time_after_wait-time_init << std::endl;

        cudaMemcpy(a_h, a_d, N*sizeof(double), cudaMemcpyDeviceToHost);
        size_t limit = 256;
        limit = N>limit ? limit : N;
        for(size_t i=N-limit; i<N; ++i)
            std::cout << a_h[i] << ((i+1)%block_dim ? " " : " | ");
        std::cout <<  std::endl;
    }

    return 0;
}

