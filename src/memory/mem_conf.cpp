#include "config.h"
#include "mem_conf.hpp"

size_t openfpm_ofpmmemory_compilation_mask()
{
	size_t compiler_mask = 0;

	#ifdef CUDA_ON_CPU
	compiler_mask |= 0x1;
	#endif

	#ifdef __NVCC__
	compiler_mask |= 0x02;
	#endif

	#ifdef CUDA_GPU
	compiler_mask |= 0x04;
	#endif

	return compiler_mask;
}