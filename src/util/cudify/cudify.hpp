#ifndef CUDIFY_HPP_
#define CUDIFY_HPP_

#ifdef CUDIFY_USE_ALPAKA
#else
#include "cudify_sequencial.hpp"
#endif

#endif