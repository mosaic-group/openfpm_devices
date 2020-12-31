#ifndef CUDIFY_HPP_
#define CUDIFY_HPP_

#ifdef CUDIFY_USE_ALPAKA
#include "cudify_alpaka.hpp"
#else
#include "cudify_sequencial.hpp"
#endif

#endif