#include <cstddef>
#include <cuda_runtime.h>
#include "CudaMemory.cuh"
#include "cuda_macro.h"

/*! \brief Allocate a chunk of memory
 *
 * Allocate a chunk of memory
 *
 * \param sz size of the chunk of memory to allocate in byte
 *
 */
bool CudaMemory::allocate(size_t sz)
{
	//! Allocate the device memory
	if (dm == NULL)
	{CUDA_SAFE_CALL(cudaAlloc(&dv,sz,cudaAlloc));}
}

/*! \brief destroy a chunk of memory
 *
 * Destroy a chunk of memory
 *
 */
void CudaMemory::destroy()
{
	CUDA_SAFE_CALL(cudaDestroy(dv));
}

/*! \brief copy the device memory to a pointer
 *
 * copy the device memory to a pointer
 *
 */
void CudaMemory::copyToPointer()
{
	// check if we have a host buffer, if not allocate it

	if (hm == NULL)
	{CUDA_SAFE_CALL(cudaHostAlloc(&hm,sz*sizeof(T),cudaHostAllocMapped))}

	// put on queue a copy from device to host

	t.call();

	// put on queue a memory copy from pointers
}

void CudaMemory::copyDeviceToDevice()
{
	// put on queue a copy from device to device

	t.call();
}

bool CudaMemory::copy(memory m)
{
	//! Here we try to cast memory into OpenFPMwdeviceCudaMemory
	CudaMemory * ofpm = dynamic_cast<CudaMemory>(m);

	//! if we fail we get the pointer and simply copy from the pointer

	if (ofpm == NULL)
	{
		// copy the memory from device to host and from host to device

		copyFromPointer(t);
	}
	else
	{
		// they are the same memory type, use cuda/thrust buffer copy

		copyDeviceToDevice();
	}
}

bool CudaMemory::copy(OpenFPMwdeviceCudaMemory m)
{
	// they are the same type of memory so copy from device to device

	copyDeviceToDevice();
}

size_t CudaMemory::size()
{
	dv->size();
}

bool CudaMemory::resize(size_t sz)
{
	//! Allocate the device memory
	if (dv == NULL)
	{dv = new boost::shared_ptr<void>(new thrust::device_vector<void>());}
	else
	{dv.get()->resize(sz);}
}
