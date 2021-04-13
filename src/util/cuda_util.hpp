/*
 * cuda_util.hpp
 *
 *  Created on: Jun 13, 2018
 *      Author: i-bird
 */

#ifndef OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_
#define OPENFPM_DATA_SRC_UTIL_CUDA_UTIL_HPP_

#include "config.h"
#if defined(__HIP__)
	// If hip fill NVCC think it is Nvidia compiler
	#ifdef __NVCC__
		#undef __NVCC__
		#include <hip/hip_runtime.h>
		#define __NVCC__
	#else
		#include <hip/hip_runtime.h>
	#endif
#elif defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
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
		#define __global__
		#endif

		#ifdef CUDA_ON_CPU 
			
			#ifdef __shared__
				#undef __shared__
			#endif
			#define __shared__ static

		#else

			#ifndef __shared__
			#define __shared__
			#endif

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
