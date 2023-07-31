#pragma once

namespace gpu {

struct cuda_exception_t : std::exception {
  cudaError_t result;

  cuda_exception_t(cudaError_t result_) : result(result_) { }
  virtual const char* what() const noexcept { 
    return cudaGetErrorString(result); 
  }
};

}
