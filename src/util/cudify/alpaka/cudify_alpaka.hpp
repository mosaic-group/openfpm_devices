#ifndef CUDIFY_ALPAKA_HPP_
#define CUDIFY_ALPAKA_HPP_

/*! \brief This file wrap CUDA functions and some CUB and MGPU function into CPU
 *
 * This file use ALPAKA as underline accelerator implementation.
 *
 * At the moment performances make it useless with mostly any available accelerator.
 *
 */

#include "util/cudify/cudify_hardware_cpu.hpp"
#include "boost/bind.hpp"
#include <type_traits>

#define CUDA_ON_BACKEND CUDA_BACKEND_ALPAKA

extern alpa_base_structs __alpa_base__;

extern thread_local dim3 threadIdx;
extern thread_local dim3 blockIdx;

extern dim3 blockDim;
extern dim3 gridDim;

static void __syncthreads()    
{
    // saving states
    dim3 threadIdx_s = threadIdx;
    dim3 blockIdx_s = blockIdx;
    dim3 blockDim_s = blockDim;
    dim3 gridDim_s = gridDim;

    alpaka::syncBlockThreads(*__alpa_base__.accKer);

    // restoring states
    threadIdx = threadIdx_s;
    blockIdx = blockIdx_s;
    blockDim = blockDim_s;
    gridDim = gridDim_s;
};

static void cudaDeviceSynchronize()
{
    alpaka::wait(*__alpa_base__.queue);
}

static void cudaMemcpyFromSymbol(void * dev_mem,const unsigned char * global_cuda_error_array,size_t sz)
{
    memcpy(dev_mem,global_cuda_error_array,sz);
}

/**
 * CUDA memory copy types
 */
enum  cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

extern int vct_atomic_add;
extern int vct_atomic_rem;

static void cudaMemcpyToSymbol(unsigned char * global_cuda_error_array,const void * mem,size_t sz,int offset,int unused)
{
    memcpy(global_cuda_error_array+offset,mem,sz);
}

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
                    context_t& gpuContext)
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
            c_keys_it c_keys, c_vals_it c_vals, comp_t comp, context_t& gpuContext)
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
{
    if (__alpa_base__.initialized == true) {return;}

    __alpa_base__.devAcc = new AccType_alpa(alpaka::getDevByIdx<Acc_alpa>(0u));

    // Create a queue on the device
    __alpa_base__.queue = new Queue_alpa(*__alpa_base__.devAcc);

    __alpa_base__.initialized = true;
}

#ifdef PRINT_CUDA_LAUNCHES

#define CUDA_LAUNCH(cuda_call,ite, ...)\
        {\
        Vec_alpa const elementsPerThread(Vec_alpa::all(static_cast<Idx_alpa>(1)));\
        Vec_alpa const grid_d((Idx_alpa)ite.wthr.x,(Idx_alpa)ite.wthr.y,(Idx_alpa)ite.wthr.z);\
        Vec_alpa const thread_d((Idx_alpa)ite.thr.x,(Idx_alpa)ite.thr.y,(Idx_alpa)ite.thr.z);\
        WorkDiv_alpa const workDiv = WorkDiv_alpa(grid_d,thread_d,elementsPerThread);\
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
        alpaka::exec<Acc_alpa>(\
        *__alpa_base__.queue,\
        workDiv,\
        [&] ALPAKA_FN_ACC(Acc_alpa const& acc) -> void {\
            \
            auto globalThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);\
            auto globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);\
            auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);\
            \
            blockIdx.x = globalBlockIdx[0];\
            blockIdx.y = globalBlockIdx[1];\
            blockIdx.z = globalBlockIdx[2];\
            \
            threadIdx.x = globalThreadIdx[0];\
            threadIdx.y = globalThreadIdx[1];\
            threadIdx.z = globalThreadIdx[2];\
            \
            __alpa_base__.accKer = &acc;\
\
            cuda_call(__VA_ARGS__);\
        });\
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }


#define CUDA_LAUNCH_DIM3(cuda_call,wthr_,thr_, ...)\
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        Vec_alpa const elementsPerThread(Vec_alpa::all(static_cast<Idx_alpa>(1)));\
        Vec_alpa const grid_d((Idx_alpa)wthr__.x,(Idx_alpa)wthr__.y,(Idx_alpa)wthr__.z);\
        Vec_alpa const thread_d((Idx_alpa)thr__.x,(Idx_alpa)thr__.y,(Idx_alpa)thr__.z);\
        WorkDiv_alpa const workDiv = WorkDiv_alpa(grid_d,thread_d,elementsPerThread);\
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
        alpaka::exec<Acc_alpa>(\
        *__alpa_base__.queue,\
        workDiv,\
        [&] ALPAKA_FN_ACC(Acc_alpa const& acc) -> void {\
            \
            auto globalThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);\
            auto globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);\
            auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);\
            \
            blockIdx.x = globalBlockIdx[0];\
            blockIdx.y = globalBlockIdx[1];\
            blockIdx.z = globalBlockIdx[2];\
            \
            threadIdx.x = globalThreadIdx[0];\
            threadIdx.y = globalThreadIdx[1];\
            threadIdx.z = globalThreadIdx[2];\
            \
            __alpa_base__.accKer = &acc;\
\
            cuda_call(__VA_ARGS__);\
        });\
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_CHECK()

#else

#define CUDA_LAUNCH(cuda_call,ite, ...)\
        {\
        Vec_alpa const elementsPerThread(Vec_alpa::all(static_cast<Idx_alpa>(1)));\
        Vec_alpa const grid_d((Idx_alpa)ite.wthr.x,(Idx_alpa)ite.wthr.y,(Idx_alpa)ite.wthr.z);\
        Vec_alpa const thread_d((Idx_alpa)ite.thr.x,(Idx_alpa)ite.thr.y,(Idx_alpa)ite.thr.z);\
        WorkDiv_alpa const workDiv = WorkDiv_alpa(grid_d,thread_d,elementsPerThread);\
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
        \
        alpaka::exec<Acc_alpa>(\
        *__alpa_base__.queue,\
        workDiv,\
        [&] ALPAKA_FN_ACC(Acc_alpa const& acc) -> void {\
            \
            auto globalThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);\
            auto globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);\
            auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);\
            \
            blockIdx.x = globalBlockIdx[0];\
            blockIdx.y = globalBlockIdx[1];\
            blockIdx.z = globalBlockIdx[2];\
            \
            threadIdx.x = globalThreadIdx[0];\
            threadIdx.y = globalThreadIdx[1];\
            threadIdx.z = globalThreadIdx[2];\
            \
            __alpa_base__.accKer = &acc;\
\
            cuda_call(__VA_ARGS__);\
        });\
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }


#define CUDA_LAUNCH_DIM3(cuda_call,wthr_,thr_, ...)\
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        Vec_alpa const elementsPerThread(Vec_alpa::all(static_cast<Idx_alpa>(1)));\
        Vec_alpa const grid_d((Idx_alpa)wthr__.x,(Idx_alpa)wthr__.y,(Idx_alpa)wthr__.z);\
        Vec_alpa const thread_d((Idx_alpa)thr__.x,(Idx_alpa)thr__.y,(Idx_alpa)thr__.z);\
        WorkDiv_alpa const workDiv = WorkDiv_alpa(grid_d,thread_d,elementsPerThread);\
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
        alpaka::exec<Acc_alpa>(\
        *__alpa_base__.queue,\
        workDiv,\
        [&] ALPAKA_FN_ACC(Acc_alpa const& acc) -> void {\
            \
            auto globalThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);\
            auto globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);\
            auto globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);\
            \
            blockIdx.x = globalBlockIdx[0];\
            blockIdx.y = globalBlockIdx[1];\
            blockIdx.z = globalBlockIdx[2];\
            \
            threadIdx.x = globalThreadIdx[0];\
            threadIdx.y = globalThreadIdx[1];\
            threadIdx.z = globalThreadIdx[2];\
            \
            __alpa_base__.accKer = &acc;\
\
            cuda_call(__VA_ARGS__);\
        });\
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_CHECK()

#endif

#endif
