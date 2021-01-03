#include "config.h"
#include "cudify_hardware_common.hpp"
#ifdef HAVE_BOOST_CONTEXT
#include <boost/context/continuation.hpp>
#endif
#include <vector>
#include "cudify_hardware_common.hpp"

#ifdef HAVE_ALPAKA
#include "cudify_hardware_alpaka.hpp"

alpa_base_structs __alpa_base__;
#endif

#ifdef CUDA_ON_CPU

dim3 threadIdx;
dim3 blockIdx;

dim3 blockDim;
dim3 gridDim;

#endif

int vct_atomic_add;
int vct_atomic_rem;

#ifdef HAVE_BOOST_CONTEXT
std::vector<void *> mem_stack;
std::vector<boost::context::detail::fcontext_t> contexts;
int cur_fib;
void * par_glob;
boost::context::detail::fcontext_t main_ctx;
#endif