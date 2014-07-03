#pragma once

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

// wrapper around cuda events
class CudaEvent {
  public:

    // constructor
    CudaEvent() {
        cudaError_t status = 
            cudaEventCreate(&event_);

        #ifdef DEBUG_MSG
        std::cout << "CudaEvent() :: " << event_ << std::endl;
        #endif
        assert(status == cudaSuccess);
    }

    // desctructor
    // no need to wait for event to finish:
    // in the case that an event has been recorded and not yet completed, cudaEventDestroy()
    // will return immediately, and the resources associated with the event will be released automatically
    // when the event finishes.
    ~CudaEvent() {

        #ifdef DEBUG_MSG
        std::cout << "~CudaEvent() :: " << event_ << std::endl;
        #endif
        cudaEventDestroy(event_);
    }

    cudaEvent_t& event() {
        return event_;
    }

    // force host execution to wait for event completion
    void wait() {
        // the event isn't actually in use, so just return
        cudaError_t status =
            cudaEventSynchronize(event_);
    }

  private:

    cudaEvent_t event_;
};

