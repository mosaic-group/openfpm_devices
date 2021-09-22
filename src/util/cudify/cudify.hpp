#ifndef CUDIFY_HPP_
#define CUDIFY_HPP_

#if defined(CUDIFY_USE_ALPAKA)
#include "cudify_alpaka.hpp"
#elif defined(CUDIFY_USE_OPENMP)
#include "cudify_openmp.hpp"
#else
#include "cudify_sequencial.hpp"
#endif

#endif