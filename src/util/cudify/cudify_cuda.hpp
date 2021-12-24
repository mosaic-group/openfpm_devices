#ifndef __CUDIFY_CUDA_HPP__
#define __CUDIFY_CUDA_HPP__

#define CUDA_ON_BACKEND CUDA_BACKEND_CUDA

constexpr int default_kernel_wg_threads_ = 1024;

#if CUDART_VERSION >= 11000 && defined(__NVCC__)
    #include "cub/util_type.cuh"
    #include "cub/block/block_scan.cuh"
#endif

#ifdef __NVCC__

template<typename lambda_f>
__global__ void kernel_launch_lambda(lambda_f f)
{
    dim3 bid = blockIdx;
    dim3 tid = threadIdx;
    f(bid,tid);
}

#endif

static void init_wrappers()
{}

#if defined(SE_CLASS1) || defined(CUDA_CHECK_LAUNCH)

#define CUDA_LAUNCH(cuda_call,ite, ...) \
        {\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
        }\
        CHECK_SE_CLASS1_PRE\
        if (ite.wthr.x != 0)\
        {cuda_call<<<ite.wthr,ite.thr>>>(__VA_ARGS__);}\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
            CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }\
        }

#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...) \
        {\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
        }\
        CHECK_SE_CLASS1_PRE\
        cuda_call<<<wthr,thr>>>(__VA_ARGS__);\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
            CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
        }\
        }

#define CUDA_LAUNCH_LAMBDA(ite, lambda_f, ...) \
        {\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
        }\
        CHECK_SE_CLASS1_PRE\
        if (ite.wthr.x != 0)\
        {kernel_launch_lambda<<<ite.wthr,ite.thr>>>(lambda_f);}\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
            CHECK_SE_CLASS1_POST("lambda")\
        }\
        }

#define CUDA_CHECK() \
        {\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda an error has occurred before, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
        }\
        CHECK_SE_CLASS1_PRE\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
            CHECK_SE_CLASS1_POST("no call","no args")\
        }\
        }

#else

#define CUDA_LAUNCH(cuda_call,ite, ...) \
        if (ite.wthr.x != 0)\
        {cuda_call<<<ite.wthr,ite.thr>>>(__VA_ARGS__);}

#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...) \
        cuda_call<<<wthr,thr>>>(__VA_ARGS__);

#define CUDA_LAUNCH_LAMBDA(ite,lambda_f, ...) \
        kernel_launch_lambda<<<ite.wthr,ite.thr>>>(lambda_f);

#define CUDA_CHECK()

#endif

#endif
