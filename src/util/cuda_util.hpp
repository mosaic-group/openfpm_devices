/*
 * cuda_util.hpp
 *
 *  Created on: Jun 13, 2018
 *      Author: i-bird
 */

#ifndef OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_
#define OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_

#include "config.h"
#if defined(__NVCC__) && !defined(CUDA_ON_CPU)
#include <cuda_runtime.h>
#endif

#ifdef CUDA_GPU

	#ifndef __NVCC__

		#ifndef __host__
		#define __host__
		#define __device__
		#define __shared__
		#define __global__
		#endif

	#else

		#ifndef __host__
		#define __host__
		#define __device__
		#define __shared__
		#define __global__
		#endif

		#ifdef CUDA_ON_CPU 

		#define CUDA_SAFE(cuda_call) \
		cuda_call;

		#else

		#define CUDA_SAFE(cuda_call) \
		cuda_call; \
		{\
			cudaError_t e = cudaPeekAtLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
		}

		#endif

	#endif
#else

#ifndef __host__
#define __host__
#define __device__
#define __shared__ static
#define __global__
#endif

#endif


#endif /* OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_ */
