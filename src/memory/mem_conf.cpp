#include "config.h"
#include "util/cuda_launch.hpp"
#include "mem_conf.hpp"

size_t openfpm_ofpmmemory_compilation_mask()
{
	size_t compiler_mask = 0;

	compiler_mask = CUDA_ON_BACKEND;

	return compiler_mask;
}