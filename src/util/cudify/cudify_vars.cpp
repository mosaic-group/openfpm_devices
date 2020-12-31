#include "config.h"
#include "cudify_hardware_common.hpp"
#include <boost/context/continuation.hpp>
#include <vector>

#ifdef HAVE_ALPAKA
#include "cudify_hardware_alpaka.hpp"

alpa_base_structs __alpa_base__;
#endif

dim3 threadIdx;
dim3 blockIdx;

dim3 blockDim;
dim3 gridDim;

int vct_atomic_add;
int vct_atomic_rem;


std::vector<void *> mem_stack;
std::vector<boost::context::detail::fcontext_t> contexts;
int cur_fib;
void * par_glob;
boost::context::detail::fcontext_t main_ctx;