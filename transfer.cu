#include <iostream>
#include <algorithm>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "CudaEvent.h"
#include "CudaStream.h"

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel that adds the vectors a[N] += b[N]
///////////////////////////////////////////////////////////////////////////////
__global__
void sum(double* a, double* b, size_t N) {
    size_t tid = threadIdx.x + blockDim.x*blockIdx.x;
    size_t grid_step = blockDim.x * gridDim.x;

    while(tid<N) {
        a[tid] += b[tid];
        tid += grid_step;
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper function that will initialize cuda.
///////////////////////////////////////////////////////////////////////////////
bool initialize_cuda() {
    CUresult result = cuInit(0);
    return result == CUDA_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// helper function for allocating device memory
///////////////////////////////////////////////////////////////////////////////
template <typename T>
T* allocate_on_device(size_t N) {
    void* ptr=0;
    cudaMalloc(&ptr, N*sizeof(T));
    return reinterpret_cast<T*>(ptr);
}

///////////////////////////////////////////////////////////////////////////////
// helper function for allocating host memory
// takes boolean flag indicating whether to use pinned memory or not
///////////////////////////////////////////////////////////////////////////////
template <typename T>
T* allocate_on_host(size_t N, T value=T(), bool pinned=false) {
    T* ptr=0;
    if( pinned ) {
        cudaHostAlloc((void**)&ptr, N*sizeof(T), cudaHostAllocPortable);
    }
    else {
        ptr = reinterpret_cast<T*>(malloc(N*sizeof(T)));
    }
    std::fill(ptr, ptr+N, value);

    return ptr;
}

// wrapper for launch configuration
class Launch {
  public:
    Launch(size_t N, unsigned int b_dim)
    : block_(b_dim)
    {
        unsigned int g_dim = N/b_dim + (N%b_dim ? 1 : 0);
        g_dim = g_dim > 1024 ? 1024 : g_dim;
        grid_ = dim3(g_dim);
    }

    dim3 block() {
        return block_;
    }

    dim3 grid() {
        return grid_;
    }

    unsigned int block_dim() {
        return block_.x;
    }

    unsigned int grid_dim() {
        return grid_.x;
    }

  private:
    Launch();

    dim3 block_;
    dim3 grid_;
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int main(void) {

    const size_t N=128*1024*1024;
    const size_t size = sizeof(double)*N;

    // initialize CUDA
    if(!initialize_cuda()) {
        std::cerr << "unable to initialize CUDA" << std::endl;
        return 1;
    }

    // create streams
    CudaStream stream_H2D(true);
    CudaStream stream_D2H(true);
    CudaStream stream_compute(true);

    // check that streams are/are not default stream as appropriate
    assert(!stream_H2D.is_default_stream());
    assert(!stream_D2H.is_default_stream());
    assert(!stream_compute.is_default_stream());

    CudaEvent event_H2D;
    CudaEvent event_compute;
    CudaEvent event_D2H;

    Launch launch(N, 128);
    std::cout << "launch grid : " << launch.block_dim() << "*" << launch.grid_dim() << std::endl;

    // allocate host and device memory
    double *a_d = allocate_on_device<double>(N);
    double *b_d = allocate_on_device<double>(N);

    double *a_h = allocate_on_host<double>(N, 1., true);
    double *b_h = allocate_on_host<double>(N, 1., true);

    // copy data to device
    cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream_H2D.stream());
    cudaMemcpyAsync(b_d, b_h, size, cudaMemcpyHostToDevice, stream_H2D.stream());

    // insert events that force compute stream to wait
    stream_H2D.insert_event(event_H2D);
    stream_compute.wait_on_event(event_H2D);

    // asynchronously execute the kernel
    sum<<<launch.block(), launch.grid(), 0, stream_compute.stream()>>>(a_d, b_d, N);

    // insert event
    stream_compute.insert_event(event_compute);
    stream_D2H.wait_on_event(event_compute);

    cudaMemcpyAsync(a_h, a_d, size, cudaMemcpyDeviceToHost, stream_D2H.stream());
    stream_D2H.insert_event(event_D2H);

    event_D2H.wait();

    size_t limit = 256;
    limit = N>limit ? limit : N;
    for(size_t i=N-limit; i<N; ++i)
        std::cout << a_h[i] << ((i+1)%launch.block_dim() ? " " : " | ");
    std::cout <<  std::endl;

    return 0;
}


