#ifndef CUDIFY_HPP_
#define CUDIFY_HPP_

#include "config.h"

#define CUDA_BACKEND_NONE 0
#define CUDA_BACKEND_CUDA 1
#define CUDA_BACKEND_SEQUENTIAL 2
#define CUDA_BACKEND_ALPAKA 3
#define CUDA_BACKEND_OPENMP 4
#define CUDA_BACKEND_HIP 5


#if defined(CUDIFY_USE_CUDA)
#include "cuda/cudify_cuda.hpp"
#elif defined(CUDIFY_USE_ALPAKA)
#include "alpaka/cudify_alpaka.hpp"
#elif defined(CUDIFY_USE_OPENMP)
#include "openmp/cudify_openmp.hpp"
#elif defined(CUDIFY_USE_HIP)
#include "hip/cudify_hip.hpp"
#elif defined(CUDIFY_USE_SEQUENTIAL)
#include "sequential/cudify_sequential.hpp"
#else
#define CUDA_ON_BACKEND CUDA_BACKEND_NONE

constexpr int default_kernel_wg_threads_ = 1024;

static void init_wrappers() {}

#endif

#endif
