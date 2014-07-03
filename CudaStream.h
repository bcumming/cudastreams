#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaEvent.h"

// wrapper around cuda streams
class CudaStream {
  public:
    // default constructor
    // sets to the default stream 0
    CudaStream() : stream_(0) {};

    // constructor with flag for whether or not to create a new stream
    // if no stream is to be created, then default stream 0 is used
    CudaStream(bool create_new_stream) {
        stream_ = 0;
        if(create_new_stream)
            stream_ = new_stream();

        #ifdef DEBUG_MSG
        std::cout << "\tCudaStream(" << (create_new_stream ? "true" : "false")
                  << ") -> " << stream_ << std::endl;
        #endif
    }

    // destructor
    ~CudaStream() {
        #ifdef DEBUG_MSG
        std::cout << "\t~CudaStream() -> " << stream_ << std::endl;
        #endif
        if(stream_) {
            cudaError_t status = cudaStreamDestroy(stream_);
            #ifdef DEBUG_MSG
            std::cout << (status==cudaSuccess ? "\t\tdestroyed" : "\t\tunable to destroy") << std::endl;
            #endif
            assert(status == cudaSuccess);
        }
    }

    bool is_default_stream() {
        return stream_==0;
    }

    // return the cuda Stream handle
    cudaStream_t stream() {
        return stream_;
    }

    // insert event into stream
    // returns immediately
    void insert_event(CudaEvent &e) {
        //cudaError_t status =
            //cudaEventRecord(e.event(), stream_);

        // set the event as being in use
        //e.lock();
        e.record(stream_);

        assert(status == cudaSuccess);
    }

    // make all future work on stream wait until event has completed.
    // returns immediately, not waiting for event to complete
    void wait_on_event(CudaEvent &e) {
        cudaError_t status =
            cudaStreamWaitEvent(stream_, e.event(), 0);

        assert(status == cudaSuccess);
    }

  private:

    // helper that creates a new CUDA stream by calling the 
    // CUDA API function cudaStreamCreate()
    cudaStream_t new_stream() {
        cudaStream_t s;
        cudaError_t status =
            cudaStreamCreate(&s);

        assert(status == cudaSuccess);

        return s;
    }

    cudaStream_t stream_;
};
