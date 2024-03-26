/*
 * cuda_util.hpp
 *
 *  Created on: Jun 13, 2018
 *      Author: i-bird
 */

#ifndef OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_
#define OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_

#include "config.h"
#include "cuda_kernel_error_checker.hpp"

#if defined(CUDIFY_USE_ALPAKA)
#define CUDA_ON_CPU
#elif defined(CUDIFY_USE_OPENMP)
#define CUDA_ON_CPU
#elif defined(CUDIFY_USE_SEQUENTIAL)
#define CUDA_ON_CPU
#endif

// CUDA_GPU: CUDA, HIP, SEQUENTIAL, OPENMP, ALPAKA
#ifdef CUDA_GPU
       #ifndef __NVCC__
		#ifndef __host__
		#define __host__
		#define __device__
		#define __forceinline__
		#define __shared__ static thread_local
		#define __global__ inline
		#endif
	#endif

	#ifdef CUDA_ON_CPU
		#ifndef __host__
		#define __host__
		#define __device__
		#define __forceinline__
		#define __global__ inline
		#endif

		#ifdef __shared__
			#undef __shared__
		#endif
		#define __shared__ static thread_local
	#endif
#else
	#ifndef __host__
	#define __host__
	#define __forceinline__
	#define __device__
	#define __shared__ static thread_local
	#define __global__ inline
	#endif
#endif

#define CUDA_BACKEND_NONE 0
#define CUDA_BACKEND_CUDA 1
#define CUDA_BACKEND_SEQUENTIAL 2
#define CUDA_BACKEND_ALPAKA 3
#define CUDA_BACKEND_OPENMP 4
#define CUDA_BACKEND_HIP 5



#if defined(CUDIFY_USE_CUDA)
#include "cudify/cuda/cudify_cuda.hpp"
#elif defined(CUDIFY_USE_ALPAKA)
#include "cudify/alpaka/cudify_alpaka.hpp"
#elif defined(CUDIFY_USE_OPENMP)
#include "cudify/openmp/cudify_openmp.hpp"
#elif defined(CUDIFY_USE_HIP)
#include "cudify/hip/cudify_hip.hpp"
#elif defined(CUDIFY_USE_SEQUENTIAL)
#include "cudify/sequential/cudify_sequential.hpp"
#else
#define CUDA_ON_BACKEND CUDA_BACKEND_NONE

constexpr int default_kernel_wg_threads_ = 1024;

static void init_wrappers() {}


#endif

#endif /* OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_ */
