#include "cudify_hardware.hpp"

alpa_base_structs __alpa_base__;

dim3 threadIdx;
dim3 blockIdx;

dim3 blockDim;
dim3 gridDim;

int vct_atomic_add;
int vct_atomic_rem;

boost::context::fiber main_fib;

std::vector<void *> mem_stack;
std::vector<boost::context::detail::fcontext_t> contexts;
int cur_fib;
void * par_glob;
boost::context::detail::fcontext_t main_ctx;