#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

// wrapper around cuda streams
class CudaStream {
    public:
    // default constructor
    // sets to the default stream 0
    CudaStream() : stream_(0) {};

    // constructor with flag for whether or not
    // not create a new stream
    // if no stream is to be created, then default stream 0 is used
    CudaStream(bool create_new_stream) {
        stream_ = 0;
        if(create_new_stream) {
            stream_ = new_stream();
        }
        std::cout << "\tCudaStream(" << (create_new_stream ? "true" : "false")
                  << ") -> " << stream_ << std::endl;
    }

    // destructor
    ~CudaStream() {
        std::cout << "\t~CudaStream() -> " << stream_ << std::endl;
        if(stream_) {
            cudaError_t status =
                cudaStreamDestroy(stream_);
            std::cout << (status==cudaSuccess ? "\t\tdestroyed" : "\t\tunable to destroy") << std::endl;
        }
    }

    bool is_default_stream() {
        return stream_==0;
    }

    cudaStream_t stream() {
        return stream_;
    }

    private:

    cudaStream_t new_stream() {
        cudaStream_t s;
        cudaError_t status =
            cudaStreamCreate(&s);

        if(status != cudaSuccess)
            return cudaStream_t(0);

        return s;
    }

    cudaStream_t stream_;
};
