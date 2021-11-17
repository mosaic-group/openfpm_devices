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
#include "util/cudify/cudify_cuda.hpp"
#elif defined(CUDIFY_USE_ALPAKA)
#include "cudify_alpaka.hpp"
#elif defined(CUDIFY_USE_OPENMP)
#include "cudify_openmp.hpp"
#elif defined(CUDIFY_USE_HIP)
#include "cudify_hip.hpp"
#elif defined(CUDIFY_USE_SEQUENTIAL)
#include "cudify_sequencial.hpp"
#else
#define CUDA_ON_BACKEND CUDA_BACKEND_NONE

constexpr int default_kernel_wg_threads_ = 1024;

static void init_wrappers() {}

#endif

#endif
