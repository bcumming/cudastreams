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
        for(int i=0; i<50; ++i)
            a[tid] += exp(1./a[tid])+b[tid];
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
        //cudaHostAlloc((void**)&ptr, N*sizeof(T), cudaHostAllocPortable);
        std::cout << "allocating " << N*sizeof(T) << " bytes pinned host data" << std::endl;
        cudaMallocHost((void**)&ptr, N*sizeof(T));
    }
    else {
        std::cout << "allocating " << N*sizeof(T) << " bytes unpinned host data" << std::endl;
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
    const size_t nchunks=1;
    const size_t chunk_dim=N/nchunks;
    const size_t size = sizeof(double)*N;
    const size_t chunk_size = size/nchunks;

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

    Launch launch(N, 128);
    std::cout << "launch grid : " << launch.block_dim() << "*" << launch.grid_dim() << std::endl;

    // allocate host and device memory
    double *a_d = allocate_on_device<double>(N);
    double *b_d = allocate_on_device<double>(N);

    double *a_h = allocate_on_host<double>(N, 1., true);
    double *b_h = allocate_on_host<double>(N, 1., true);

    CudaEvent event_start = stream_H2D.insert_event();

    // copy data to device
    for(int i=0; i<nchunks; ++i) {
        size_t offset = i*chunk_dim;
        cudaMemcpyAsync(a_d+offset, a_h+offset, chunk_size, cudaMemcpyHostToDevice, stream_H2D.stream());
        cudaMemcpyAsync(b_d+offset, b_h+offset, chunk_size, cudaMemcpyHostToDevice, stream_H2D.stream());

        // insert events that force compute stream to wait
        CudaEvent event_H2D = stream_H2D.insert_event();
        stream_compute.wait_on_event(event_H2D);

        // asynchronously execute the kernel
        sum<<<launch.block(), launch.grid(), 0, stream_compute.stream()>>>(a_d+offset, b_d+offset, chunk_dim);

        // insert event
        CudaEvent event_compute =
            stream_compute.insert_event();
        stream_D2H.wait_on_event(event_compute);

        cudaMemcpyAsync(a_h+offset, a_d+offset, chunk_size, cudaMemcpyDeviceToHost, stream_D2H.stream());
        CudaEvent event_D2H =
            stream_D2H.insert_event();
    }

    CudaEvent event_end = stream_D2H.insert_event();
    event_end.wait();
    double time_taken = event_end.time_since(event_start);
    std::cout << "that took " << time_taken << " seconds" << std::endl;

    /*
    size_t limit = 256;
    limit = N>limit ? limit : N;
    double result = 0.;
    for(size_t i=N-limit; i<N; ++i)
        std::cout << a_h[i] << ((i+1)%launch.block_dim() ? " " : " | ");
    #pragma omp parallel for reduction(+:result)
    for(size_t i=0; i<N; ++i)
        result += 2. - a_h[i];
    std::cout << std::endl;

    std::cout << "result : " << result << std::endl;
    */

    return 0;
}


