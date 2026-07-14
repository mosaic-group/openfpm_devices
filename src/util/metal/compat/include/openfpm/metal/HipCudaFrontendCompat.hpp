#ifndef OPENFPM_HIP_CUDA_FRONTEND_COMPAT_HPP
#define OPENFPM_HIP_CUDA_FRONTEND_COMPAT_HPP

// Existing OpenFPM .cu files use __NVCC__ as a legacy "GPU compiler" guard.
// chipStar's HIP headers interpret that macro as the NVIDIA platform if it is
// visible during platform selection.  Select the SPIR-V HIP platform first,
// then restore the compatibility macro before the original source is parsed.
#if defined(CUDIFY_USE_METAL) && defined(__HIPCC__) && defined(__NVCC__)
#undef __NVCC__
#include <hip/hip_runtime.h>
#define __NVCC__ 1
#endif

#endif
