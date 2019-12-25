/*
 * cuda_macro.h
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

#include <iostream>

#ifdef __HIPCC__

#define CUDA_SAFE_CALL(call) {\
        hipError_t err = call;\
        if (hipSuccess != err) {\
                std::cerr << "Cuda error in file "<< __FILE__ << " in line " << __LINE__ <<  ": " << hipGetErrorString(err);\
        }\
}

#else

#define CUDA_SAFE_CALL(call) {\
	cudaError_t err = call;\
	if (cudaSuccess != err) {\
		std::cerr << "Cuda error in file "<< __FILE__ << " in line " << __LINE__ <<  ": " << cudaGetErrorString(err);\
	}\
}

#endif

