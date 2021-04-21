/*
 * cuda_launch.hpp
 *
 *  Created on: Jan 14, 2019
 *      Author: i-bird
 */

#ifndef CUDA_LAUNCH_HPP_
#define CUDA_LAUNCH_HPP_

#include "config.h"
#include "cuda_kernel_error_checker.hpp"

#if defined(__NVCC__) && !defined(CUDA_ON_CPU) && !defined(__HIP__)

	constexpr int default_kernel_wg_threads_ = 1024;

	#include "cub/util_type.cuh"
	#include "cub/block/block_scan.cuh"

	#if defined(SE_CLASS1) || defined(CUDA_CHECK_LAUNCH)

	#define CUDA_LAUNCH(cuda_call,ite, ...) \
			{\
			cudaDeviceSynchronize(); \
			{\
				cudaError_t e = cudaGetLastError();\
				if (e != cudaSuccess)\
				{\
					std::string error = cudaGetErrorString(e);\
					std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
				}\
			}\
			CHECK_SE_CLASS1_PRE\
			if (ite.wthr.x != 0)\
			{cuda_call<<<ite.wthr,ite.thr>>>(__VA_ARGS__);}\
			cudaDeviceSynchronize(); \
			{\
				cudaError_t e = cudaGetLastError();\
				if (e != cudaSuccess)\
				{\
					std::string error = cudaGetErrorString(e);\
					std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
				}\
				CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
			}\
			}

	#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...) \
			{\
			cudaDeviceSynchronize(); \
			{\
				cudaError_t e = cudaGetLastError();\
				if (e != cudaSuccess)\
				{\
					std::string error = cudaGetErrorString(e);\
					std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
				}\
			}\
			CHECK_SE_CLASS1_PRE\
			cuda_call<<<wthr,thr>>>(__VA_ARGS__);\
			cudaDeviceSynchronize(); \
			{\
				cudaError_t e = cudaGetLastError();\
				if (e != cudaSuccess)\
				{\
					std::string error = cudaGetErrorString(e);\
					std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
				}\
				CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
			}\
			}

	#define CUDA_CHECK() \
			{\
			cudaDeviceSynchronize(); \
			{\
				cudaError_t e = cudaGetLastError();\
				if (e != cudaSuccess)\
				{\
					std::string error = cudaGetErrorString(e);\
					std::cout << "Cuda an error has occurred before, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
				}\
			}\
			CHECK_SE_CLASS1_PRE\
			cudaDeviceSynchronize(); \
			{\
				cudaError_t e = cudaGetLastError();\
				if (e != cudaSuccess)\
				{\
					std::string error = cudaGetErrorString(e);\
					std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
				}\
				CHECK_SE_CLASS1_POST("no call","no args")\
			}\
			}

	#else

	#define CUDA_LAUNCH(cuda_call,ite, ...) \
			if (ite.wthr.x != 0)\
			{cuda_call<<<ite.wthr,ite.thr>>>(__VA_ARGS__);}

	#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...) \
			cuda_call<<<wthr,thr>>>(__VA_ARGS__);

	#define CUDA_CHECK()

	#endif

#else

#include "util/cudify/cudify.hpp"

#endif

#endif /* CUDA_LAUNCH_HPP_ */
