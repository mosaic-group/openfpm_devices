#ifndef CUDIFY_HPP_
#define CUDIFY_HPP_

#ifdef CUDIFY_USE_ALPAKA
#include "cudify_alpaka.hpp"
#elif defined(CUDIFY_USE_HIP)
#include "cudify_hip.hpp"
#else
#include "cudify_sequencial.hpp"
#endif

#endif