#ifndef CUDIFY_HIP_HPP_
#define CUDIFY_HIP_HPP_

#include "config.h"

#define CUDA_ON_BACKEND CUDA_BACKEND_HIP

#ifdef __NVCC__
    #undef __NVCC__
    #include <hip/hip_runtime.h>
    #define __NVCC__
#else
    #include <hip/hip_runtime.h>
#endif

constexpr int default_kernel_wg_threads_ = 256;

typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipDeviceProp_t cudaDeviceProp_t;
typedef cudaDeviceProp_t cudaDeviceProp;
typedef hipEvent_t cudaEvent_t;
typedef hipFuncAttributes cudaFuncAttributes;


#define cudaSuccess hipSuccess


static void init_wrappers()
{}

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

static cudaError_t cudaMemcpyToSymbol(unsigned char * global_cuda_error_array,const void * mem,size_t sz,int offset,cudaMemcpyKind opt)
{
    hipMemcpyKind opt_;

    switch (opt)
    {
        case cudaMemcpyHostToHost:
            opt_ = hipMemcpyHostToHost;
            break;

        case cudaMemcpyHostToDevice:
            opt_ = hipMemcpyHostToDevice;
            break;

        case cudaMemcpyDeviceToHost:
            opt_ = hipMemcpyDeviceToHost;
            break;

        case cudaMemcpyDeviceToDevice:
            opt_ = hipMemcpyDeviceToDevice;
            break;

        default:
            opt_ = hipMemcpyDefault;
            break;
    }

    return hipMemcpyToSymbol(global_cuda_error_array,mem,sz,offset,opt_);
}

static cudaError_t cudaDeviceSynchronize()
{
    return hipDeviceSynchronize();
}

static cudaError_t cudaMemcpyFromSymbol(void * dev_mem,const unsigned char * global_cuda_error_array,size_t sz)
{
    return hipMemcpyFromSymbol(dev_mem,global_cuda_error_array,sz);
}

static const char* cudaGetErrorString ( cudaError_t error )
{
    return hipGetErrorString(error);
}

static cudaError_t cudaGetDevice ( int* device )
{
    return hipGetDevice(device);
}

static cudaError_t cudaSetDevice ( int  device )
{
    return hipSetDevice(device);
}

static cudaError_t cudaMemGetInfo ( size_t* free, size_t* total )
{
    return hipMemGetInfo(free,total);
}

static cudaError_t cudaFuncGetAttributes ( cudaFuncAttributes* attr, const void* func )
{
    return hipFuncGetAttributes(attr,func);
}

static cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )
{
    return hipGetDeviceProperties(prop,device);
}

static cudaError_t cudaEventCreate ( cudaEvent_t* event )
{
    return hipEventCreate(event);
}

static cudaError_t cudaEventDestroy ( cudaEvent_t event )
{
    return hipEventDestroy(event);
}

static cudaError_t cudaMalloc ( void** devPtr, size_t size )
{
    return hipMalloc(devPtr,size);
}

static cudaError_t cudaMallocHost ( void** ptr, size_t size )
{
    return hipHostMalloc(ptr,size);
}

static cudaError_t cudaFree ( void* devPtr )
{
    return hipFree(devPtr);
}

static cudaError_t cudaFreeHost ( void* ptr )
{
    return hipHostFree(ptr);
}

static cudaError_t cudaStreamSynchronize ( cudaStream_t stream )
{
    return hipStreamSynchronize(stream);
}

static cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )
{
    return hipEventRecord(event,stream);
}

static cudaError_t cudaEventSynchronize ( cudaEvent_t event )
{
    return hipEventSynchronize(event);
}

static cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )
{
    return hipEventElapsedTime(ms,start,end);
}

static cudaError_t cudaGetDeviceCount ( int* count )
{
    return hipGetDeviceCount(count);
}

static cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind opt )
{
    hipMemcpyKind opt_;

    switch (opt)
    {
        case cudaMemcpyHostToHost:
            opt_ = hipMemcpyHostToHost;
            break;

        case cudaMemcpyHostToDevice:
            opt_ = hipMemcpyHostToDevice;
            break;

        case cudaMemcpyDeviceToHost:
            opt_ = hipMemcpyDeviceToHost;
            break;

        case cudaMemcpyDeviceToDevice:
            opt_ = hipMemcpyDeviceToDevice;
            break;

        default:
            opt_ = hipMemcpyDefault;
            break;
    }

    return hipMemcpy(dst,src,count,opt_);
}

#ifdef __HIPCC__

#include "cudify_hardware_common.hpp"
#include "util/cuda_util.hpp"
#include <vector>
#include <string.h>
#include "hipcub/hipcub.hpp"
#include "hipcub/block/block_scan.hpp"

template<typename lambda_f>
__global__ void kernel_launch_lambda(lambda_f f)
{
    dim3 bid = blockIdx;
    dim3 tid = threadIdx;
    f(bid,tid);
}

template<typename lambda_f>
__global__ void kernel_launch_lambda_tls(lambda_f f)
{
    f();
}

namespace cub
{
    template<typename T, unsigned int bd>
    using BlockScan = hipcub::BlockScan<T,bd>;
}

template<typename T>
struct has_work_gpu_cl_lin_blocks_
{
    static unsigned int lin(const T & b)
    {
        return b.x * b.y * b.z;
    }
};

template<>
struct has_work_gpu_cl_lin_blocks_<unsigned int>
{
    static unsigned int lin(const unsigned int & b)
    {
        return b;
    }
};

template<>
struct has_work_gpu_cl_lin_blocks_<unsigned long>
{
    static unsigned int lin(const unsigned long & b)
    {
        return b;
    }
};

template<>
struct has_work_gpu_cl_lin_blocks_<int>
{
    static unsigned int lin(const int & b)
    {
        return b;
    }
};

template<typename wthr_type, typename thr_type>
bool has_work_gpu_cl_(const wthr_type & wthr, const thr_type & thr)
{
    return (has_work_gpu_cl_lin_blocks_<typename std::remove_const<wthr_type>::type>::lin(wthr) * 
            has_work_gpu_cl_lin_blocks_<typename std::remove_const<thr_type>::type>::lin(thr)) != 0;
}

#ifdef PRINT_CUDA_LAUNCHES

#define CUDA_LAUNCH(cuda_call,ite, ...)\
        \
        CHECK_SE_CLASS1_PRE\
        \
        std::cout << "Launching: " << #cuda_call << std::endl;\
        \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(cuda_call), dim3(ite.wthr), dim3(ite.thr), 0, 0, __VA_ARGS__);\
        \
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
        CHECK_SE_CLASS1_PRE\
        std::cout << "Launching: " << #cuda_call << std::endl;\
        \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(cuda_call), dim3(ite.wthr), dim3(ite.thr), 0, 0, __VA_ARGS__);\
        \
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_CHECK()

#else

#define CUDA_LAUNCH(cuda_call,ite, ...) \
        \
        {\
        CHECK_SE_CLASS1_PRE\
        \
        if (has_work_gpu_cl_(ite.wthr,ite.thr)  == true)\
        {hipLaunchKernelGGL(HIP_KERNEL_NAME(cuda_call), dim3(ite.wthr), dim3(ite.thr), 0, 0, __VA_ARGS__);}\
        \
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }


#define CUDA_LAUNCH_DIM3(cuda_call,wthr_,thr_, ...)\
        {\
        \
        CHECK_SE_CLASS1_PRE\
        \
        if (has_work_gpu_cl_(wthr_,thr_) == true)\
        {hipLaunchKernelGGL(HIP_KERNEL_NAME(cuda_call),wthr_,thr_, 0, 0, __VA_ARGS__);}\
        \
        CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }

#define CUDA_LAUNCH_LAMBDA(ite,lambda_f, ...)\
        {\
        \
        CHECK_SE_CLASS1_PRE\
        \
        if (has_work_gpu_cl_(ite.wthr,ite.thr) == true)\
        {hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda),ite.wthr,ite.thr, 0, 0, lambda_f);}\
        \
        CHECK_SE_CLASS1_POST("kernel_launch_lambda",__VA_ARGS__)\
        }

#define CUDA_LAUNCH_LAMBDA_TLS(ite, lambda_f, ...) \
        {\
        CHECK_SE_CLASS1_PRE\
        if (ite.wthr.x != 0)\
        {hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda_tls),ite.wthr,ite.thr,0,0,lambda_f);}\
	CHECK_SE_CLASS1_POST("kernel_launch_lambda",__VA_ARGS__)\
        }

#define CUDA_LAUNCH_LAMBDA_DIM3_TLS(wthr_,thr_, lambda_f, ...) \
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
	CHECK_SE_CLASS1_PRE\
        if (wthr__.x != 0)\
        {hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda_tls),wthr_,thr_, 0, 0, lambda_f);}\
	CHECK_SE_CLASS1_POST("kernel_launch_lambda",__VA_ARGS__)\
        }

#define CUDA_LAUNCH_LAMBDA_DIM3(wthr_,thr_, lambda_f, ...) \
        {\
        dim3 wthr__(wthr_);\
        dim3 thr__(thr_);\
        CHECK_SE_CLASS1_PRE\
        if (wthr__.x != 0)\
        {hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda),wthr_,thr_, 0, 0, lambda_f);}\
        CHECK_SE_CLASS1_POST("kernel_launch_lambda",__VA_ARGS__)\
        }

#define CUDA_CHECK()

#endif

#endif


#endif
