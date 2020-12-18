#ifndef CUDIFY_HPP_
#define CUDIFY_HPP_

#include "cudify_hardware.hpp"
#include "cuda_util.hpp"
#include "boost/bind.hpp"

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

            T cur = tmp[0];
            tmp[0] = 0;
            for (int i = 1 ; i < dim ; i++)
            {
                tmp[i] = tmp[i-1] + cur;
            }
        }
    };
}

namespace mgpu
{
    template<typename input_it,
             typename segments_it, typename output_it, typename op_t, typename type_t, typename context_t>
    void segreduce(input_it input, int count, segments_it segments,
                    int num_segments, output_it output, op_t op, type_t init,
                    context_t& context)
    {
        for (int i = 0 ; i < num_segments - 1; i++)
        {
            output[i] = 0;
            for (int j = segments[i] ; j < segments[i+1] ; j++)
            {
                op(output[i],input[j]);
            }
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

        while (a_it < a_count && b_it < b_count)
        {
            if (comp(a_keys[a_it],b_keys[b_it]))
            {
                c_keys[c_it] = a_keys[a_it];
                c_vals[c_it] = a_vals[a_it];
                c_it++;
                a_it++;
            }
            else
            {
                c_keys[c_it] = b_keys[a_it] + a_count;
                c_vals[c_it] = b_vals[a_it];
                c_it++;
                b_it++;
            }
        }
    }
}

template<typename T, typename T2>
static T atomicAdd(T * address, T2 val)  
{
    T old = *address;
    *address += val;
    return old;
};

#define MGPU_HOST_DEVICE

namespace mgpu
{
    template<typename type_t>
    struct less_t : public std::binary_function<type_t, type_t, bool> {
    bool operator()(type_t a, type_t b) const {
        return a < b;
    }
    };
/*    template<typename type_t>
    struct less_equal_t : public std::binary_function<type_t, type_t, bool> {
    MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
        return a <= b;
    }
    };*/
    template<typename type_t>
    struct greater_t : public std::binary_function<type_t, type_t, bool> {
    MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
        return a > b;
    }
    };
/*    template<typename type_t>
    struct greater_equal_t : public std::binary_function<type_t, type_t, bool> {
    MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
        return a >= b;
    }
    };
    template<typename type_t>
    struct equal_to_t : public std::binary_function<type_t, type_t, bool> {
    MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
        return a == b;
    }
    };
    template<typename type_t>
    struct not_equal_to_t : public std::binary_function<type_t, type_t, bool> {
    MGPU_HOST_DEVICE bool operator()(type_t a, type_t b) const {
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
        MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
        return a - b;
    }
    };

    template<typename type_t>
    struct multiplies_t : public std::binary_function<type_t, type_t, type_t> {
    MGPU_HOST_DEVICE type_t operator()(type_t a, type_t b) const {
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

static void init_alpaka()
{
    if (__alpa_base__.initialized == true) {return;}

    __alpa_base__.devAcc = new AccType_alpa(alpaka::getDevByIdx<Acc_alpa>(0u));

    // Create a queue on the device
    __alpa_base__.queue = new Queue_alpa(*__alpa_base__.devAcc);

    __alpa_base__.initialized = true;
}

#define CUDA_LAUNCH(cuda_call,ite, ...)\
        {\
        Vec_alpa const elementsPerThread(Vec_alpa::all(static_cast<Idx_alpa>(1)));\
        Vec_alpa const grid_d((Idx_alpa)ite.wthr.x,(Idx_alpa)ite.wthr.y,(Idx_alpa)ite.wthr.z);\
        Vec_alpa const thread_d((Idx_alpa)ite.thr.x,(Idx_alpa)ite.thr.y,(Idx_alpa)ite.thr.z);\
        WorkDiv_alpa const workDiv = WorkDiv_alpa(grid_d,thread_d,elementsPerThread);\
        \
        gridDim.x = ite.wthr.x * ite.thr.x;\
        gridDim.y = ite.wthr.y * ite.thr.y;\
        gridDim.z = ite.wthr.z * ite.thr.z;\
        \
        blockDim.x = ite.thr.x;\
        blockDim.y = ite.thr.y;\
        blockDim.z = ite.thr.z;\
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
        }


#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...)
    
/*    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerGrid(Vec::all(static_cast<Idx>(8)));
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);*/

#define CUDA_CHECK()

#endif