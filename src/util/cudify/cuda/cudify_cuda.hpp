#ifndef __CUDIFY_CUDA_HPP__
#define __CUDIFY_CUDA_HPP__

#define CUDA_ON_BACKEND CUDA_BACKEND_CUDA
#include <cuda_runtime.h>
#include <boost/preprocessor.hpp>

#ifdef DEFAULT_CUDA_THREADS
constexpr size_t default_kernel_wg_threads_ = static_cast<size_t>(DEFAULT_CUDA_THREADS);
#else
constexpr size_t default_kernel_wg_threads_ = static_cast<size_t>(1024);
#endif

#if CUDART_VERSION >= 11000 && defined(__NVCC__)
    #include "cub/util_type.cuh"
    #include "cub/block/block_scan.cuh"
#endif

#ifdef __NVCC__
#include "operators.hpp"

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


/**
 * @brief Find appropriate grid and block size based on statistics of register usage during compilation
 * @note
 * - According to https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/:
 * This can greatly simplify the task of frameworks (such as Thrust), that must launch user-defined kernels. This is
 * also handy for kernels that are not primary performance bottlenecks, where the programmer just wants a simple way
 * to run the kernel with correct results, rather than hand-tuning the execution configuration.
 *
 * -For advanced kernel hand-tuning depending on compute capability, the launchbox feature of moderngpu
 * (https://moderngpu.github.io/performance.html) should be considered.
 */
template<typename dim3Type, typename... Args>
void FixConfigLaunch(void (* _kernel)(Args...), dim3Type & wthr, dim3Type & thr) {

        if (thr.x != 0xFFFFFFFF) {
            return;
        }

        int blockSize = 0; // The launch configurator returned block size
        int minGridSize;   // The minimum grid size needed to achieve the
                           // maximum occupancy for a full device launch

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, *_kernel, 0, 0);

        int dim = (wthr.x != 0) + (wthr.y != 0) + (wthr.z != 0);
        if (dim == 0) {
            return;
        }

	    size_t tot_work;

        unsigned int wthr_x = wthr.x;
        unsigned int wthr_y = wthr.y;
        unsigned int wthr_z = wthr.z;
        
        if (dim == 1)
            tot_work = wthr.x;
        else if (dim == 2)
            tot_work = wthr.x * wthr.y;
        else if (dim == 3)
            tot_work = wthr.x * wthr.y * wthr.z;

        // round to the nearest bigger power of 2
        size_t tot_work_2 = tot_work; 
		tot_work_2--;
		tot_work_2 |= tot_work_2 >> 1;   
		tot_work_2 |= tot_work_2 >> 2;
		tot_work_2 |= tot_work_2 >> 4;
		tot_work_2 |= tot_work_2 >> 8;
		tot_work_2 |= tot_work_2 >> 16;
		tot_work_2++;

	    size_t n = (tot_work <= blockSize)?tot_work_2:blockSize;

	    if (tot_work == 0)
	    {
		    thr.x = 0;
		    thr.y = 0;
		    thr.z = 0;

		    wthr.x = 0;
		    wthr.y = 0;
		    wthr.z = 0;
	    }

	    thr.x = 1;
	    thr.y = 1;
	    thr.z = 1;

	    int dir = 0;

	    while (n != 1)
	    {
		    if (dir % 3 == 0)
		    {thr.x = thr.x << 1;}
		    else if (dir % 3 == 1)
		    {thr.y = thr.y << 1;}
		    else if (dir % 3 == 2)
		    {thr.z = thr.z << 1;}

		    n = n >> 1;

		    dir++;
		    dir %= dim;
	    }

	    if (dim >= 1)
	    {wthr.x = (wthr.x) / thr.x + (((wthr_x)%thr.x != 0)?1:0);}


	    if (dim >= 2)
	    {wthr.y = (wthr.y) / thr.y + (((wthr_y)%thr.y != 0)?1:0);}
	    else
	    {wthr.y = 1;}

	    if (dim >= 3)
	    {wthr.z = (wthr.z) / thr.z + (((wthr_z)%thr.z != 0)?1:0);}
	    else
	    {wthr.z = 1;}

	// crop if wthr == 1

	if (dim >= 1 && wthr.x == 1)
	{thr.x = wthr_x;}

	if (dim >= 2 && wthr.y == 1)
	{thr.y = wthr_y;}

	if (dim == 3 && wthr.z == 1)
	{thr.z = wthr_z;}
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

#define CUDA_LAUNCH_DIM3_DEBUG_SE1(cuda_call,wthr,thr, ...) \
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
            CHECK_SE_CLASS1_POST("lambda",0)\
        }\
        }

#define CUDA_LAUNCH_LAMBDA_TLS(ite, lambda_f, ...) \
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
            CHECK_SE_CLASS1_POST("lambda",0)\
        }\
        }

#define CUDA_LAUNCH_LAMBDA_DIM3_TLS(wthr_,thr_, lambda_f, ...) \
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
        {kernel_launch_lambda<<<wthr_,thr_>>>(lambda_f);}\
        cudaDeviceSynchronize(); \
        {\
            cudaError_t e = cudaGetLastError();\
            if (e != cudaSuccess)\
            {\
                std::string error = cudaGetErrorString(e);\
                std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
            }\
            CHECK_SE_CLASS1_POST("lambda",0)\
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

template<typename... Args, typename ite_type>
void CUDA_LAUNCH(void (* _kernel)(Args...),ite_type ite,Args... args)
{
//    std::cout << "DEMANGLE " << typeid(decltype(_kernel)).name() << " " << ite.wthr.x << " " << ite.wthr.y << " " << ite.wthr.z << "/" << ite.thr.x << " " << ite.thr.y << " " << ite.thr.z  << std::endl;

    #ifdef __NVCC__
    FixConfigLaunch(_kernel,ite.wthr,ite.thr);
    _kernel<<<ite.wthr,ite.thr>>>(args...);
    #else
    std::cout << __FILE__ << ":" << __LINE__ << " " << "CUDA_LAUNCH not implemented for this compiler" << std::endl;
    #endif
}

template<typename... Args>
void CUDA_LAUNCH_DIM3(void (* _kernel)(Args...),dim3 wthr, dim3 thr,Args... args)
{
//    std::cout << "DEMANGLE " << typeid(decltype(_kernel)).name() << "  " << wthr.x << " " << wthr.y << " " << wthr.z << "/" << thr.x << " " << thr.y << " " << thr.z  << std::endl;

    #ifdef __NVCC__
    FixConfigLaunch(_kernel,wthr,thr);
    _kernel<<<wthr,thr>>>(args...);
    #else
    std::cout << __FILE__ << ":" << __LINE__ << " " << "CUDA_LAUNCH_DIM3 not implemented for this compiler" << std::endl;
    #endif
}

template<typename lambda_type, typename ite_type, typename... Args>
void CUDA_LAUNCH_LAMBDA(ite_type ite, lambda_type lambda_f, Args... args)
{
    #ifdef __NVCC__
    void (* _ker)(lambda_type) = kernel_launch_lambda;
    FixConfigLaunch(_ker,ite.wthr,ite.thr);
    
    kernel_launch_lambda<<<ite.wthr,ite.thr>>>(lambda_f);
    #else
    std::cout << __FILE__ << ":" << __LINE__ << " " << "CUDA_LAUNCH_LAMBDA not implemented for this compiler" << std::endl;
    #endif
}

static void CUDA_CHECK() {}

template<typename lambda_type, typename ite_type, typename... Args>
void CUDA_LAUNCH_LAMBDA_TLS(ite_type ite, lambda_type lambda_f, Args... args)
{
    #ifdef __NVCC__
    void (* _ker)(lambda_type) = kernel_launch_lambda;
    FixConfigLaunch(_ker,ite.wthr,ite.thr);

    if (ite.wthr.x != 0)
    {kernel_launch_lambda<<<ite.wthr,ite.thr>>>(lambda_f);}
    #else
    std::cout << __FILE__ << ":" << __LINE__ << " " << "CUDA_LAUNCH_LAMBDA_TLS not implemented for this compiler" << std::endl;
    #endif
}

template<typename lambda_type, typename... Args>
void CUDA_LAUNCH_LAMBDA_DIM3(dim3 wthr_, dim3 thr_, lambda_type lambda_f, Args... args)
{
    #ifdef __NVCC__
    void (* _ker)(lambda_type) = kernel_launch_lambda;
    FixConfigLaunch(_ker,wthr_,thr_);

    dim3 wthr__(wthr_);
    dim3 thr__(thr_);
    if (wthr__.x != 0)
    {kernel_launch_lambda<<<wthr__,thr__>>>(lambda_f);}
    #else
    std::cout << __FILE__ << ":" << __LINE__ << " " << "CUDA_LAUNCH_LAMBDA_TLS not implemented for this compiler" << std::endl;
    #endif
}

template<typename lambda_type, typename... Args>
void CUDA_LAUNCH_LAMBDA_DIM3_TLS(dim3 wthr_, dim3 thr_, lambda_type lambda_f, Args... args)
{
    #ifdef __NVCC__
    void (* _ker)(lambda_type) = kernel_launch_lambda;
    FixConfigLaunch(_ker,wthr_,thr_);

    dim3 wthr__(wthr_);
    dim3 thr__(thr_);
    if (wthr__.x != 0)
    {kernel_launch_lambda<<<wthr__,thr__>>>(lambda_f);}
    #else
    std::cout << __FILE__ << ":" << __LINE__ << " " << "CUDA_LAUNCH_LAMBDA_TLS not implemented for this compiler" << std::endl;
    #endif
}

#endif

#endif
