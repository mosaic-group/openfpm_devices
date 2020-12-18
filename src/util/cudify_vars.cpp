
#include "cudify_hardware.hpp"

alpa_base_structs __alpa_base__;

thread_local dim3 threadIdx;
thread_local dim3 blockIdx;

dim3 blockDim;
dim3 gridDim;
