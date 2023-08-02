#include "config.h"
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <type_traits>

std::is_trivially_copyable<int> b;

#include "util/cuda_launch.hpp"
#include "memory/CudaMemory.cuh"

#ifdef CUDIFY_USE_OPENMP
BOOST_AUTO_TEST_SUITE( cudify_tests_openmp )
#elif defined(CUDIFY_USE_CUDA)
BOOST_AUTO_TEST_SUITE( cudify_tests_cuda )
#elif defined(CUDIFY_USE_HIP)
BOOST_AUTO_TEST_SUITE( cudify_tests_hip )
#else
BOOST_AUTO_TEST_SUITE( cudify_tests_sequential )
#endif

struct par_struct
{
    float * ptr;
};

struct ite_g
{
    dim3 wthr;
    dim3 thr;

    size_t nblocks()
	{
		return wthr.x * wthr.y * wthr.z;
	}

	size_t nthrs()
	{
		return thr.x * thr.y * thr.z;
	}
};

template<typename T>
__global__ void test1(float * array,T p)
{
    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    array[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = 5.0;

    p.ptr[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = 17.0;
}

template<typename T>
__global__ void test1_syncthreads(T p, float * array)
{
    __shared__ int cnt;

    cnt = 0;

    __syncthreads();

    size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    array[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = 5.0;

    p.ptr[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = 17.0;

    atomicAdd(&cnt,1);

    __syncthreads();

    array[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = cnt;

    __syncthreads();

    atomicAdd(&cnt,1);

    __syncthreads();

    p.ptr[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = cnt;
}


BOOST_AUTO_TEST_CASE( cudify_on_test_test )
{
    init_wrappers();

    CudaMemory mem;
    mem.allocate(16*16*16*sizeof(float));

    CudaMemory mem2;
    mem2.allocate(16*16*16*sizeof(float));

    float * array_ptr = (float *)mem.getDevicePointer();

    par_struct p;
    p.ptr = (float *)mem2.getDevicePointer();

    ite_g g;

    g.wthr = dim3(4,4,4);
    g.thr = dim3(4,4,4);

    CUDA_LAUNCH(test1,g,array_ptr,p);

    mem.deviceToHost();
    mem2.deviceToHost();

    float * ptr1 = (float *)mem.getPointer();
    float * ptr2 = (float *)mem2.getPointer();

    bool check = true;
    for (int i = 0 ; i < 16*16*16; i++)
    {
        check &= ptr1[i] == 5.0;
        check &= ptr2[i] == 17.0;
    }

    BOOST_REQUIRE_EQUAL(check,true);
}

BOOST_AUTO_TEST_CASE( cudify_on_test_test2)
{
    init_wrappers();

    CudaMemory mem;
    mem.allocate(16*16*16*sizeof(float));

    CudaMemory mem2;
    mem2.allocate(16*16*16*sizeof(float));

    float * array_ptr = (float *)mem.getDevicePointer();

    par_struct p;
    p.ptr = (float *)mem2.getDevicePointer();

    ite_g g;

    g.wthr = dim3(4,4,4);
    g.thr = dim3(4,4,4);

    CUDA_LAUNCH(test1_syncthreads,g,p,array_ptr);

    mem.deviceToHost();
    mem2.deviceToHost();

    float * ptr1 = (float *)mem.getPointer();
    float * ptr2 = (float *)mem2.getPointer();

    bool check = true;
    for (int i = 0 ; i < 16*16*16; i++)
    {
        //std::cout << i << "   " << ptr1[i] << "  " << ptr2[i] << std::endl;

        check &= ptr1[i] == 64.0;
        check &= ptr2[i] == 128.0;
    }

    BOOST_REQUIRE_EQUAL(check,true);
}


BOOST_AUTO_TEST_CASE( cudify_on_test_test2_lambda)
{
    init_wrappers();

    CudaMemory mem;
    mem.allocate(16*16*16*sizeof(float));

    CudaMemory mem2;
    mem2.allocate(16*16*16*sizeof(float));

    float * array_ptr = (float *)mem.getDevicePointer();

    par_struct p;
    p.ptr = (float *)mem2.getDevicePointer();

    ite_g g;

    float * array = array_ptr;

    g.wthr = dim3(4,4,4);
    g.thr = dim3(4,4,4);

    auto lambda_f = [array,p] __device__ (dim3 & blockIdx, dim3 & threadIdx){
        __shared__ int cnt;

        cnt = 0;
    
        __syncthreads();
    
        size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    
        array[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = 5.0;
    
        p.ptr[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = 17.0;
    
        atomicAdd(&cnt,1);
    
        __syncthreads();
    
        array[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = cnt;
    
        __syncthreads();
    
        atomicAdd(&cnt,1);
    
        __syncthreads();
    
        p.ptr[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = cnt;
    };

    CUDA_LAUNCH_LAMBDA(g, lambda_f);

    mem.deviceToHost();
    mem2.deviceToHost();

    float * ptr1 = (float *)mem.getPointer();
    float * ptr2 = (float *)mem2.getPointer();

    bool check = true;
    for (int i = 0 ; i < 16*16*16; i++)
    {
        //std::cout << i << "   " << ptr1[i] << "  " << ptr2[i] << std::endl;

        check &= ptr1[i] == 64.0;
        check &= ptr2[i] == 128.0;
    }

    BOOST_REQUIRE_EQUAL(check,true);
}

BOOST_AUTO_TEST_CASE( cudify_type_chack)
{
    float3 f3;

    f3.x = 0.0;
    f3.y = 1.0;
    f3.z = 2.0;

    BOOST_REQUIRE_EQUAL(f3.x,0.0);
    BOOST_REQUIRE_EQUAL(f3.y,1.0);
    BOOST_REQUIRE_EQUAL(f3.z,2.0);

    float4 f4 = make_float4(0.0,1.0,2.0,3.0);


    BOOST_REQUIRE_EQUAL(f4.x,0.0);
    BOOST_REQUIRE_EQUAL(f4.y,1.0);
    BOOST_REQUIRE_EQUAL(f4.z,2.0);
    BOOST_REQUIRE_EQUAL(f4.w,3.0);
}

BOOST_AUTO_TEST_SUITE_END()

