#ifndef CUDIFY_SEQUENCIAL_HPP_
#define CUDIFY_SEQUENCIAL_HPP_

#define CUDA_ON_BACKEND CUDA_BACKEND_SEQUENTIAL

#include "config.h"

constexpr int default_kernel_wg_threads_ = 1024;

#include "util/cudify/cudify_hardware_cpu.hpp"

#ifdef HAVE_BOOST_CONTEXT

#include "util/cuda_util.hpp"
#include <boost/bind/bind.hpp>
#include <type_traits>
#ifdef HAVE_BOOST_CONTEXT
#include <boost/context/continuation.hpp>
#endif
#include <vector>
#include <string.h>


#ifndef CUDIFY_BOOST_CONTEXT_STACK_SIZE
#define CUDIFY_BOOST_CONTEXT_STACK_SIZE 8192
#endif

extern std::vector<void *>mem_stack;

extern thread_local dim3 threadIdx;
extern thread_local dim3 blockIdx;

static dim3 blockDim;
static dim3 gridDim;

extern std::vector<void *> mem_stack;
extern std::vector<boost::context::detail::fcontext_t> contexts;
extern thread_local void * par_glob;
extern thread_local boost::context::detail::fcontext_t main_ctx;

static void __syncthreads()
{
    boost::context::detail::jump_fcontext(main_ctx,par_glob);
};



extern int thread_local vct_atomic_add;
extern int thread_local vct_atomic_rem;


namespace cub
{
    template<typename T, unsigned int dim>
    class BlockScan
    {
        public: 
        typedef std::array<T,dim> TempStorage;

        private:
        TempStorage & tmp;

        public:

        

        BlockScan(TempStorage & tmp)
        :tmp(tmp)
        {};

        void ExclusiveSum(T & in, T & out)
        {
            tmp[threadIdx.x] = in;

            __syncthreads();

            if (threadIdx.x == 0)
            {
                T prec = tmp[0];
                tmp[0] = 0;
                for (int i = 1 ; i < dim ; i++)
                {
                    auto next = tmp[i-1] + prec;
                    prec = tmp[i];
                    tmp[i] = next;
                }
            }

            __syncthreads();

            out = tmp[threadIdx.x];
            return;
        }
    };
}


template<typename T, typename T2>
static T atomicAdd(T * address, T2 val)  
{
    T old = *address;
    *address += val;
    return old;
};

namespace gpu
{
    template<typename type_t>
    struct less_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a < b;
    }
    template<typename type2_t, typename type3_t>
    bool operator()(type2_t a, type3_t b) const {
        return a < b;
    }
    };
/*    template<typename type_t>
    struct less_equal_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a <= b;
    }
    };*/
    template<typename type_t>
    struct greater_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a > b;
    }
    template<typename type2_t, typename type3_t>
    bool operator()(type2_t a, type3_t b) const {
        return a > b;
    }
    };
/*    template<typename type_t>
    struct greater_equal_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a >= b;
    }
    };
    template<typename type_t>
    struct equal_to_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a == b;
    }
    };
    template<typename type_t>
    struct not_equal_to_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a != b;
    }
    };*/

    ////////////////////////////////////////////////////////////////////////////////
    // Device-side arithmetic operators.

    template<typename type_t>
    struct plus_t : public std::binary_function<type_t, type_t, type_t> {
        type_t operator()(type_t a, type_t b) const {
        return a + b;
    }
    };

/*    template<typename type_t>
    struct minus_t : public std::binary_function<type_t, type_t, type_t> {
        type_t operator()(type_t a, type_t b) const {
        return a - b;
    }
    };

    template<typename type_t>
    struct multiplies_t : public std::binary_function<type_t, type_t, type_t> {
    type_t operator()(type_t a, type_t b) const {
        return a * b;
    }
    };*/

    template<typename type_t>
    struct maximum_t  : public std::binary_function<type_t, type_t, type_t> {
    type_t operator()(type_t a, type_t b) const {
        return std::max(a, b);
    }
    };

    template<typename type_t>
    struct minimum_t  : public std::binary_function<type_t, type_t, type_t> {
    type_t operator()(type_t a, type_t b) const {
        return std::min(a, b);
    }
    };
}


namespace gpu
{
    template<typename input_it,
             typename segments_it, typename output_it, typename op_t, typename type_t, typename context_t>
    void segreduce(input_it input, int count, segments_it segments,
                    int num_segments, output_it output, op_t op, type_t init,
                    context_t& context)
    {
        int i = 0;
        for ( ; i < num_segments - 1; i++)
        {
            int j = segments[i];
            output[i] = input[j];
            ++j;
            for ( ; j < segments[i+1] ; j++)
            {
                output[i] = op(output[i],input[j]);
            }
        }

        // Last segment
        int j = segments[i];
        output[i] = input[j];
        ++j;
        for ( ; j < count ; j++)
        {
            output[i] = op(output[i],input[j]);
        }
    }

    // Key-value merge.
    template<typename a_keys_it, typename a_vals_it,
             typename b_keys_it, typename b_vals_it,
             typename c_keys_it, typename c_vals_it,
             typename comp_t, typename context_t>
    void merge(a_keys_it a_keys, a_vals_it a_vals, int a_count,
               b_keys_it b_keys, b_vals_it b_vals, int b_count,
            c_keys_it c_keys, c_vals_it c_vals, comp_t comp, context_t& context) 
    {
        int a_it = 0;
        int b_it = 0;
        int c_it = 0;

        while (a_it < a_count || b_it < b_count)
        {
            if (a_it < a_count)
            {
                if (b_it < b_count)
                {
                    if (comp(b_keys[b_it],a_keys[a_it]))
                    {
                        c_keys[c_it] = b_keys[b_it];
                        c_vals[c_it] = b_vals[b_it];
                        c_it++;
                        b_it++;
                    }
                    else
                    {
                        c_keys[c_it] = a_keys[a_it];
                        c_vals[c_it] = a_vals[a_it];
                        c_it++;
                        a_it++;
                    }
                }
                else
                {
                    c_keys[c_it] = a_keys[a_it];
                    c_vals[c_it] = a_vals[a_it];
                    c_it++;
                    a_it++;
                }
            }
            else
            {
                c_keys[c_it] = b_keys[b_it];
                c_vals[c_it] = b_vals[b_it];
                c_it++;
                b_it++;
            }
        }
    }
}

static void init_wrappers()
{}

template<typename lambda_f>
struct Fun_enc
{
    lambda_f Fn;

    Fun_enc(lambda_f Fn)
    :Fn(Fn)
    {}

    void run()
    {
        Fn();
    }
};

template<typename lambda_f>
struct Fun_enc_bt
{
    lambda_f Fn;
    dim3 & blockIdx;
    dim3 & threadIdx;

    Fun_enc_bt(lambda_f Fn,dim3 & blockIdx,dim3 & threadIdx)
    :Fn(Fn),blockIdx(blockIdx),threadIdx(threadIdx)
    {}

    void run()
    {
        Fn(blockIdx,threadIdx);
    }
};

template<typename Fun_enc_type>
void launch_kernel(boost::context::detail::transfer_t par)
{
    main_ctx = par.fctx;
    par_glob = par.data;
    Fun_enc_type * ptr = (Fun_enc_type *)par.data;

    ptr->run();

    boost::context::detail::jump_fcontext(par.fctx,0);
}

template<typename lambda_f, typename ite_type>
static void exe_kernel(lambda_f f, ite_type & ite)
{
    if (ite.nthrs() == 0 || ite.nblocks() == 0) {return;}

    if (mem_stack.size() < ite.nthrs())
    {
        int old_size = mem_stack.size();
        mem_stack.resize(ite.nthrs());

        for (int i = old_size ; i < mem_stack.size() ; i++)
        {
            mem_stack[i] = new char [CUDIFY_BOOST_CONTEXT_STACK_SIZE];
        }
    }

    // Resize contexts
    contexts.resize(mem_stack.size());

    Fun_enc<lambda_f> fe(f);

    for (int i = 0 ; i < ite.wthr.z ; i++)
    {
        blockIdx.z = i;
        for (int j = 0 ; j < ite.wthr.y ; j++)
        {
            blockIdx.y = j;
            for (int k = 0 ; k < ite.wthr.x ; k++)
            {
                blockIdx.x = k;
                int nc = 0;
                for (int it = 0 ; it < ite.thr.z ; it++)
                {
                    for (int jt = 0 ; jt < ite.thr.y ; jt++)
                    {
                        for (int kt = 0 ; kt < ite.thr.x ; kt++)
                        {
                            contexts[nc] = boost::context::detail::make_fcontext((char *)mem_stack[nc]+CUDIFY_BOOST_CONTEXT_STACK_SIZE-16,CUDIFY_BOOST_CONTEXT_STACK_SIZE,launch_kernel<Fun_enc<lambda_f>>);
                            nc++;
                        }
                    }
                }

                bool work_to_do = true;
                while(work_to_do)
                {
                    nc = 0;
                    // Work threads
                    for (int it = 0 ; it < ite.thr.z ; it++)
                    {
                        threadIdx.z = it;
                        for (int jt = 0 ; jt < ite.thr.y ; jt++)
                        {
                            threadIdx.y = jt;
                            for (int kt = 0 ; kt < ite.thr.x ; kt++)
                            {
                                threadIdx.x = kt;
                                auto t = boost::context::detail::jump_fcontext(contexts[nc],&fe);
                                contexts[nc] = t.fctx;
                                work_to_do &= (t.data != 0);
                                nc++;
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename lambda_f, typename ite_type>
static void exe_kernel_lambda(lambda_f f, ite_type & ite)
{
    if (ite.nthrs() == 0 || ite.nblocks() == 0) {return;}

    if (mem_stack.size() < ite.nthrs())
    {
        int old_size = mem_stack.size();
        mem_stack.resize(ite.nthrs());

        for (int i = old_size ; i < mem_stack.size() ; i++)
        {
            mem_stack[i] = new char [CUDIFY_BOOST_CONTEXT_STACK_SIZE];
        }
    }

    // Resize contexts
    contexts.resize(mem_stack.size());

    bool is_sync_free = true;

    bool first_block = true;

    for (int i = 0 ; i < ite.wthr.z ; i++)
    {
        for (int j = 0 ; j < ite.wthr.y ; j++)
        {
            for (int k = 0 ; k < ite.wthr.x ; k++)
            {
                dim3 blockIdx;
                dim3 threadIdx;
                Fun_enc_bt<lambda_f> fe(f,blockIdx,threadIdx);
                if (first_block == true || is_sync_free == false)
                {
                    blockIdx.z = i;
                    blockIdx.y = j;
                    blockIdx.x = k;
                    int nc = 0;
                    for (int it = 0 ; it < ite.thr.z ; it++)
                    {
                        for (int jt = 0 ; jt < ite.thr.y ; jt++)
                        {
                            for (int kt = 0 ; kt < ite.thr.x ; kt++)
                            {
                                contexts[nc] = boost::context::detail::make_fcontext((char *)mem_stack[nc]+CUDIFY_BOOST_CONTEXT_STACK_SIZE-16,CUDIFY_BOOST_CONTEXT_STACK_SIZE,launch_kernel<Fun_enc_bt<lambda_f>>);

                                nc++;
                            }
                        }
                    }

                    bool work_to_do = true;
                    while(work_to_do)
                    {
                        nc = 0;
                        // Work threads
                        for (int it = 0 ; it < ite.thr.z ; it++)
                        {
                            threadIdx.z = it;
                            for (int jt = 0 ; jt < ite.thr.y ; jt++)
                            {
                                threadIdx.y = jt;
                                for (int kt = 0 ; kt < ite.thr.x ; kt++)
                                {
                                    threadIdx.x = kt;
                                    auto t = boost::context::detail::jump_fcontext(contexts[nc],&fe);
                                    contexts[nc] = t.fctx;

                                    work_to_do &= (t.data != 0);
                                    is_sync_free &= !(work_to_do);
                                    nc++;
                                }
                            }
                        }
                    }
                }
                else
                {
                    blockIdx.z = i;
                    blockIdx.y = j;
                    blockIdx.x = k;
                    int fb = 0;
                    // Work threads
                    for (int it = 0 ; it < ite.thr.z ; it++)
                    {
                        threadIdx.z = it;
                        for (int jt = 0 ; jt < ite.thr.y ; jt++)
                        {
                            threadIdx.y = jt;
                            for (int kt = 0 ; kt < ite.thr.x ; kt++)
                            {
                                threadIdx.x = kt;
                                f(blockIdx,threadIdx);
                            }
                        }
                    }
                }

                first_block = false;
            }
        }
    }
}

template<typename lambda_f, typename ite_type>
static void exe_kernel_lambda_tls(lambda_f f, ite_type & ite)
{
    if (ite.nthrs() == 0 || ite.nblocks() == 0) {return;}

    if (mem_stack.size() < ite.nthrs())
    {
        int old_size = mem_stack.size();
        mem_stack.resize(ite.nthrs());

        for (int i = old_size ; i < mem_stack.size() ; i++)
        {
            mem_stack[i] = new char [CUDIFY_BOOST_CONTEXT_STACK_SIZE];
        }
    }

    // Resize contexts
    contexts.resize(mem_stack.size());

    bool is_sync_free = true;

    bool first_block = true;

    for (int i = 0 ; i < ite.wthr.z ; i++)
    {
        for (int j = 0 ; j < ite.wthr.y ; j++)
        {
            for (int k = 0 ; k < ite.wthr.x ; k++)
            {
                Fun_enc<lambda_f> fe(f);
                if (first_block == true || is_sync_free == false)
                {
                    blockIdx.z = i;
                    blockIdx.y = j;
                    blockIdx.x = k;
                    int nc = 0;
                    for (int it = 0 ; it < ite.thr.z ; it++)
                    {
                        for (int jt = 0 ; jt < ite.thr.y ; jt++)
                        {
                            for (int kt = 0 ; kt < ite.thr.x ; kt++)
                            {
                                contexts[nc] = boost::context::detail::make_fcontext((char *)mem_stack[nc]+CUDIFY_BOOST_CONTEXT_STACK_SIZE-16,CUDIFY_BOOST_CONTEXT_STACK_SIZE,launch_kernel<Fun_enc<lambda_f>>);

                                nc++;
                            }
                        }
                    }

                    bool work_to_do = true;
                    while(work_to_do)
                    {
                        nc = 0;
                        // Work threads
                        for (int it = 0 ; it < ite.thr.z ; it++)
                        {
                            threadIdx.z = it;
                            for (int jt = 0 ; jt < ite.thr.y ; jt++)
                            {
                                threadIdx.y = jt;
                                for (int kt = 0 ; kt < ite.thr.x ; kt++)
                                {
                                    threadIdx.x = kt;
                                    auto t = boost::context::detail::jump_fcontext(contexts[nc],&fe);
                                    contexts[nc] = t.fctx;

                                    work_to_do &= (t.data != 0);
                                    is_sync_free &= !(work_to_do);
                                    nc++;
                                }
                            }
                        }
                    }
                }
                else
                {
                    blockIdx.z = i;
                    blockIdx.y = j;
                    blockIdx.x = k;
                    int fb = 0;
                    // Work threads
                    for (int it = 0 ; it < ite.thr.z ; it++)
                    {
                        threadIdx.z = it;
                        for (int jt = 0 ; jt < ite.thr.y ; jt++)
                        {
                            threadIdx.y = jt;
                            for (int kt = 0 ; kt < ite.thr.x ; kt++)
                            {
                                threadIdx.x = kt;
                                f();
                            }
                        }
                    }
                }

                first_block = false;
            }
        }
    }
}

template<typename lambda_f, typename ite_type>
static void exe_kernel_no_sync(lambda_f f, ite_type & ite)
{
    for (int i = 0 ; i < ite.wthr.z ; i++)
    {
        blockIdx.z = i;
        for (int j = 0 ; j < ite.wthr.y ; j++)
        {
            blockIdx.y = j;
            for (int k = 0 ; k < ite.wthr.x ; k++)
            {
                blockIdx.x = k;
                int fb = 0;
                // Work threads
                for (int it = 0 ; it < ite.wthr.z ; it++)
                {
                    threadIdx.z = it;
                    for (int jt = 0 ; jt < ite.wthr.y ; jt++)
                    {
                        threadIdx.y = jt;
                        for (int kt = 0 ; kt < ite.wthr.x ; kt++)
                        {
                            threadIdx.x = kt;
                            f();
                        }
                    }
                }
            }
        }
    }
}

#ifdef PRINT_CUDA_LAUNCHES

#define CUDA_LAUNCH(cuda_call,ite, ...)\
        \
        gridDim.x = ite.wthr.x;\
        gridDim.y = ite.wthr.y;\
        gridDim.z = ite.wthr.z;\
        \
        blockDim.x = ite.thr.x;\
        blockDim.y = ite.thr.y;\
        blockDim.z = ite.thr.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        std::cout << "Launching: " << #cuda_call << std::endl;\
        \
        exe_kernel(\
        [&](boost::context::fiber && main) -> void {\
            \
            \
\
            cuda_call(__VA_ARGS__);\
        },ite);\
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }


#define CUDA_LAUNCH_DIM3(cuda_call,wthr_,thr_, ...)\
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        \
        ite_gpu<1> itg;\
        itg.wthr = wthr;\
        itg.thr = thr;\
        \
        gridDim.x = wthr__.x;\
        gridDim.y = wthr__.y;\
        gridDim.z = wthr__.z;\
        \
        blockDim.x = thr__.x;\
        blockDim.y = thr__.y;\
        blockDim.z = thr__.z;\
        \
        CHECK_SE_CLASS1_PRE\
        std::cout << "Launching: " << #cuda_call << std::endl;\
        \
        exe_kernel(\
        [&] (boost::context::fiber && main) -> void {\
            \
            \
\
            cuda_call(__VA_ARGS__);\
            \
            \
        });\
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_CHECK()

#else

#define CUDA_LAUNCH(cuda_call,ite, ...) \
        {\
        gridDim.x = ite.wthr.x;\
        gridDim.y = ite.wthr.y;\
        gridDim.z = ite.wthr.z;\
        \
        blockDim.x = ite.thr.x;\
        blockDim.y = ite.thr.y;\
        blockDim.z = ite.thr.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        exe_kernel([&]() -> void {\
        \
            \
            cuda_call(__VA_ARGS__);\
            \
            },ite);\
        \
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_LAUNCH_LAMBDA(ite,lambda_f) \
        {\
        gridDim.x = ite.wthr.x;\
        gridDim.y = ite.wthr.y;\
        gridDim.z = ite.wthr.z;\
        \
        blockDim.x = ite.thr.x;\
        blockDim.y = ite.thr.y;\
        blockDim.z = ite.thr.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        exe_kernel_lambda(lambda_f,ite);\
        \
        CHECK_SE_CLASS1_POST("lambda",0)\
        }

#define CUDA_LAUNCH_LAMBDA_TLS(ite,lambda_f) \
        {\
        gridDim.x = ite.wthr.x;\
        gridDim.y = ite.wthr.y;\
        gridDim.z = ite.wthr.z;\
        \
        blockDim.x = ite.thr.x;\
        blockDim.y = ite.thr.y;\
        blockDim.z = ite.thr.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        exe_kernel_lambda_tls(lambda_f,ite);\
        \
        CHECK_SE_CLASS1_POST("lambda",0)\
        }

#define CUDA_LAUNCH_DIM3(cuda_call,wthr_,thr_, ...)\
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        \
        ite_gpu<1> itg;\
        itg.wthr = wthr_;\
        itg.thr = thr_;\
        \
        gridDim.x = wthr__.x;\
        gridDim.y = wthr__.y;\
        gridDim.z = wthr__.z;\
        \
        blockDim.x = thr__.x;\
        blockDim.y = thr__.y;\
        blockDim.z = thr__.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        exe_kernel([&]() -> void {\
            \
            cuda_call(__VA_ARGS__);\
            \
            },itg);\
        \
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_LAUNCH_LAMBDA_DIM3_TLS(wthr_,thr_,lambda_f) \
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        \
        ite_gpu<1> itg;\
        itg.wthr = wthr_;\
        itg.thr = thr_;\
        gridDim.x = itg.wthr.x;\
        gridDim.y = itg.wthr.y;\
        gridDim.z = itg.wthr.z;\
        \
        blockDim.x = itg.thr.x;\
        blockDim.y = itg.thr.y;\
        blockDim.z = itg.thr.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        exe_kernel_lambda_tls(lambda_f,itg);\
        \
        }

#define CUDA_LAUNCH_DIM3_DEBUG_SE1(cuda_call,wthr_,thr_, ...)\
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        \
        ite_gpu<1> itg;\
        itg.wthr = wthr_;\
        itg.thr = thr_;\
        \
        gridDim.x = wthr__.x;\
        gridDim.y = wthr__.y;\
        gridDim.z = wthr__.z;\
        \
        blockDim.x = thr__.x;\
        blockDim.y = thr__.y;\
        blockDim.z = thr__.z;\
        \
        CHECK_SE_CLASS1_PRE\
        \
        exe_kernel([&]() -> void {\
            \
            cuda_call(__VA_ARGS__);\
            \
            },itg);\
        \
        }

#define CUDA_CHECK()

#endif

#endif

#else

constexpr int default_kernel_wg_threads_ = 1024;

#endif /* CUDIFY_SEQUENCIAL_HPP_ */
