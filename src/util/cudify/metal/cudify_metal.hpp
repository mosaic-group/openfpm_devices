#ifndef CUDIFY_METAL_HPP_
#define CUDIFY_METAL_HPP_

#define CUDA_ON_BACKEND CUDA_BACKEND_METAL

#include "../cuda/operators.hpp"

#if defined(__HIPCC__)

#include <hip/hip_runtime.h>

constexpr int default_kernel_wg_threads_ = 256;
static inline void init_wrappers() {}

using cudaError_t = hipError_t;
constexpr cudaError_t cudaSuccess = hipSuccess;
using cudaMemcpyKind = hipMemcpyKind;
constexpr cudaMemcpyKind cudaMemcpyHostToHost = hipMemcpyHostToHost;
constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr cudaMemcpyKind cudaMemcpyDefault = hipMemcpyDefault;

static inline cudaError_t cudaMemcpy(void * destination, const void * source,
	std::size_t size, cudaMemcpyKind kind)
{
	return hipMemcpy(destination,source,size,kind);
}

static inline cudaError_t cudaMemset(void * destination, int value,
	std::size_t size)
{
	return hipMemset(destination,value,size);
}

static inline cudaError_t cudaDeviceSynchronize()
{
	return hipDeviceSynchronize();
}
static inline void CUDA_CHECK() {}

// hipLaunchKernelGGL's generated stub does not expose hipLaunchKernel's return
// value to existing OpenFPM call sites.  Keep CUDA_LAUNCH source-compatible and
// turn a recorded MoltenVK pipeline/dispatch error into a host exception.
extern "C" void openfpmMetalCheckLastLaunch();

// chipStar's lean HIP headers do not ship hipCUB.  The OpenFPM kernels use the
// native backend's cub::BlockScan surface in two places, both as an integer
// exclusive sum.  Keep that source API and implement the collective directly
// with HIP shared memory and barriers so the same kernels remain translatable.
namespace cub
{
template<typename T, unsigned int block_size>
class BlockScan
{
public:
	struct TempStorage
	{
		T values[block_size];
	};

private:
	TempStorage & storage_;

public:
	__device__ explicit BlockScan(TempStorage & storage) : storage_(storage) {}

	__device__ void ExclusiveSum(T input, T & output)
	{
		const unsigned int lane = threadIdx.x;
		storage_.values[lane] = input;
		__syncthreads();
		for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1)
		{
			const T value = storage_.values[lane];
			const T addend = lane >= offset ? storage_.values[lane-offset] : T(0);
			__syncthreads();
			// Every lane must execute the same store/barrier sequence.  Leaving
			// the store conditional lets SPIR-V-to-MSL split the following
			// barrier across divergent branches; on Apple GPUs that loses
			// visibility when a scan crosses a SIMD-group boundary.
			storage_.values[lane] = value + addend;
			__syncthreads();
		}
		output = lane == 0 ? T(0) : storage_.values[lane-1];
		__syncthreads();
	}
};
}

// Keep the HIP launch surface intact in translated translation units.  These
// wrapper kernels are the same ones used by the native HIP backend, so lambda
// based Grid/SparseGrid code is emitted as an ordinary HIP kernel and follows
// the same registration and SPIR-V ABI path as a named __global__ kernel.
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

template<typename T>
struct metal_has_work_gpu_cl_lin_blocks_
{
	static unsigned int lin(const T & blocks)
	{
		return blocks.x * blocks.y * blocks.z;
	}
};

template<>
struct metal_has_work_gpu_cl_lin_blocks_<unsigned int>
{
	static unsigned int lin(const unsigned int & blocks) { return blocks; }
};

template<>
struct metal_has_work_gpu_cl_lin_blocks_<unsigned long>
{
	static unsigned int lin(const unsigned long & blocks)
	{
		return static_cast<unsigned int>(blocks);
	}
};

template<>
struct metal_has_work_gpu_cl_lin_blocks_<int>
{
	static unsigned int lin(const int & blocks)
	{
		return static_cast<unsigned int>(blocks);
	}
};

template<typename wthr_type, typename thr_type>
inline bool metal_has_work_gpu_cl_(const wthr_type & wthr,
	const thr_type & thr)
{
	using workgroups_type = typename std::remove_const<wthr_type>::type;
	using threads_type = typename std::remove_const<thr_type>::type;
	return metal_has_work_gpu_cl_lin_blocks_<workgroups_type>::lin(wthr) != 0 &&
		metal_has_work_gpu_cl_lin_blocks_<threads_type>::lin(thr) != 0;
}

#define CUDA_LAUNCH(cuda_call, ite, ...) \
	do { \
		if (metal_has_work_gpu_cl_((ite).wthr,(ite).thr)) { \
			hipLaunchKernelGGL(HIP_KERNEL_NAME(cuda_call), (ite).wthr, (ite).thr, \
				0, 0, __VA_ARGS__); \
			openfpmMetalCheckLastLaunch(); \
		} \
	} while (false)
#define CUDA_LAUNCH_DIM3(cuda_call, wthr, thr, ...) \
	do { \
		if (metal_has_work_gpu_cl_(wthr,thr)) { \
			hipLaunchKernelGGL(HIP_KERNEL_NAME(cuda_call), wthr, thr, 0, 0, \
				__VA_ARGS__); \
			openfpmMetalCheckLastLaunch(); \
		} \
	} while (false)

#define CUDA_LAUNCH_LAMBDA(ite, lambda_f, ...) \
	do { \
		if (metal_has_work_gpu_cl_((ite).wthr,(ite).thr)) { \
			hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda), \
				(ite).wthr, (ite).thr, 0, 0, lambda_f); \
			openfpmMetalCheckLastLaunch(); \
		} \
	} while (false)

#define CUDA_LAUNCH_LAMBDA_TLS(ite, lambda_f, ...) \
	do { \
		if (metal_has_work_gpu_cl_((ite).wthr,(ite).thr)) { \
			hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda_tls), \
				(ite).wthr, (ite).thr, 0, 0, lambda_f); \
			openfpmMetalCheckLastLaunch(); \
		} \
	} while (false)

#define CUDA_LAUNCH_LAMBDA_DIM3(wthr, thr, lambda_f, ...) \
	do { \
		if (metal_has_work_gpu_cl_(wthr,thr)) { \
			hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda), \
				wthr, thr, 0, 0, lambda_f); \
			openfpmMetalCheckLastLaunch(); \
		} \
	} while (false)

#define CUDA_LAUNCH_LAMBDA_DIM3_TLS(wthr, thr, lambda_f, ...) \
	do { \
		if (metal_has_work_gpu_cl_(wthr,thr)) { \
			hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_launch_lambda_tls), \
				wthr, thr, 0, 0, lambda_f); \
			openfpmMetalCheckLastLaunch(); \
		} \
	} while (false)

#else

#include "util/metal/MoltenVKContext.hpp"
#include "util/metal/MoltenVKKernel.hpp"

#include <cstring>

constexpr int default_kernel_wg_threads_ = 256;

// Host-side launch geometry compatibility. This does not make CUDA kernels
// launchable on Metal; it only lets shared iterator/configuration types retain
// their existing shape.
struct dim3
{
	unsigned int x, y, z;
	constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
	: x(x_), y(y_), z(z_) {}
};

// Kernel bodies are parsed by the host compiler as ordinary inline template
// functions.  These placeholders are never executed; the launch macro below
// dispatches the embedded SPIR-V module instead.
static const dim3 blockDim;
static const dim3 blockIdx;
static const dim3 threadIdx;

static void init_wrappers() {}
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind
{
	cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
	cudaMemcpyDeviceToDevice, cudaMemcpyDefault
};
extern "C" int hipMemcpy(void * destination, const void * source,
	std::size_t size, int kind);
extern "C" int hipMemset(void * destination, int value, std::size_t size);
static const char * cudaGetErrorString(cudaError_t) { return "Metal compatibility error"; }
static cudaError_t cudaMemcpy(void * destination, const void * source,
	std::size_t size, cudaMemcpyKind kind)
{
	return hipMemcpy(destination,source,size,static_cast<int>(kind));
}
static cudaError_t cudaMemset(void * destination, int value,
	std::size_t size)
{
	return hipMemset(destination,value,size);
}
static cudaError_t cudaHostGetDevicePointer(void ** device_pointer,
	void * host_pointer, unsigned int) { *device_pointer = host_pointer; return cudaSuccess; }
static cudaError_t cudaDeviceSynchronize() { metal_synchronize(); return cudaSuccess; }
static void CUDA_CHECK() {}

#define CUDA_LAUNCH(cuda_call, ite, ...) \
	::openfpm::metal::launch_registered_function((cuda_call), \
		::openfpm::metal::to_extent((ite).wthr), \
		::openfpm::metal::to_extent((ite).thr), \
		__VA_ARGS__)

#define CUDA_LAUNCH_DIM3(cuda_call, wthr_, thr_, ...) \
	::openfpm::metal::launch_registered_function((cuda_call), \
		::openfpm::metal::to_extent(wthr_), \
		::openfpm::metal::to_extent(thr_), \
		__VA_ARGS__)

#define CUDA_LAUNCH_LAMBDA(...) \
	static_assert(false, "CUDA_LAUNCH_LAMBDA is not implemented by the MoltenVK bridge")
#define CUDA_LAUNCH_LAMBDA_TLS(...) \
	static_assert(false, "CUDA_LAUNCH_LAMBDA_TLS requires a HIP-translated Metal translation unit")
#define CUDA_LAUNCH_LAMBDA_DIM3(...) \
	static_assert(false, "CUDA_LAUNCH_LAMBDA_DIM3 requires a HIP-translated Metal translation unit")
#define CUDA_LAUNCH_LAMBDA_DIM3_TLS(...) \
	static_assert(false, "CUDA_LAUNCH_LAMBDA_DIM3_TLS requires a HIP-translated Metal translation unit")

#endif // __HIPCC__

#endif
