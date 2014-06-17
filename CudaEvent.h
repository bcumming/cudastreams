#pragma once

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaStream.h"

// wrapper around cuda events
class CudaEvent {
    public:

    // constructor
    CudaEvent() : set_(false) {
        cudaError_t status = 
            cudaEventCreate(&event_);

        assert(status == cudaSuccess);
    }

    // constructor that will create and insert in a stream
    CudaEvent(CudaStream& s) : set_(false) {
        insert_in_stream(s);
    }

    // desctructor
    ~CudaEvent() {
        cudaEventDestroy(event_);
    }

    // insert the event into a stream
    CudaEvent& insert_in_stream(CudaStream &stream) {
        cudaError_t status =
            cudaEventRecord(event_, stream.stream());
        if(status == cudaSuccess) {
            set_ = true;
        }

        return *this;
    }

    void wait() {
        cudaError_t status =
            cudaEventSynchronize(event_);
        set_ = false;
    }

    private:

    cudaEvent_t event_;
    bool set_;
};

