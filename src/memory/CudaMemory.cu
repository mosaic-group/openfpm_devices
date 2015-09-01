#include "config.h"
#include <cstddef>
#include <cuda_runtime.h>
#include "CudaMemory.cuh"
#include "cuda_macro.h"
#include <cstring>

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
	{CUDA_SAFE_CALL(cudaMalloc(&dm,sz));}

	this->sz = sz;

	return true;
}

/*! \brief destroy a chunk of memory
 *
 * Destroy a chunk of memory
 *
 */
void CudaMemory::destroy()
{
	if (dm != NULL)
	{
		//! Release the allocated memory
		CUDA_SAFE_CALL(cudaFree(dm));
		dm = NULL;
	}

	if (hm != NULL)
	{
		//! we invalidate hm
		CUDA_SAFE_CALL(cudaFreeHost(hm));
#ifdef SE_CLASS2
		//! remove hm
		check_delete(hm);
#endif
		hm = NULL;
	}
}

/*! \brief Allocate the host buffer
 *
 * Allocate the host buffer
 *
 */

void CudaMemory::allocate_host(size_t sz)
{
	if (hm == NULL)
	{
		CUDA_SAFE_CALL(cudaHostAlloc(&hm,sz,cudaHostAllocMapped))
#ifdef SE_CLASS2
		//! add hm to the list of allocated memory
		check_new(hm,sz);
#endif
	}
}

/*! \brief copy the data from a pointer
 *
 * copy the data from a pointer
 *
 *	\param ptr
 *	\return true if success
 */
bool CudaMemory::copyFromPointer(void * ptr)
{
	// check if we have a host buffer, if not allocate it

	allocate_host(sz);

	// get the device pointer

	void * dvp;
	CUDA_SAFE_CALL(cudaHostGetDevicePointer(&dvp,hm,0));

	// memory copy

	memcpy(ptr,dvp,sz);

	return true;
}

/*! \brief copy from device to device
 *
 * copy a piece of memory from device to device
 *
 * \param CudaMemory from where to copy
 *
 * \return true is success
 */
bool CudaMemory::copyDeviceToDevice(CudaMemory & m)
{
	//! The source buffer is too big to copy it

	if (m.sz > sz)
	{
		std::cerr << "Error " << __LINE__ << __FILE__ << ": source buffer is too big to copy";
		return false;
	}

	//! Copy the memory
	CUDA_SAFE_CALL(cudaMemcpy(m.dm,dm,m.sz,cudaMemcpyDeviceToDevice));

	return true;
}

/*! \brief copy from memory
 *
 * copy from memory
 *
 * \param m a memory interface
 *
 */
bool CudaMemory::copy(memory & m)
{
	//! Here we try to cast memory into OpenFPMwdeviceCudaMemory
	CudaMemory * ofpm = dynamic_cast<CudaMemory *>(&m);

	//! if we fail we get the pointer and simply copy from the pointer

	if (ofpm == NULL)
	{
		// copy the memory from device to host and from host to device

		return copyFromPointer(m.getPointer());
	}
	else
	{
		// they are the same memory type, use cuda/thrust buffer copy

		return copyDeviceToDevice(*ofpm);
	}
}

/*! \brief Get the size of the allocated memory
 *
 * Get the size of the allocated memory
 *
 * \return the size of the allocated memory
 *
 */

size_t CudaMemory::size()
{
	return sz;
}

/*! \brief Resize the allocated memory
 *
 * Resize the allocated memory, if request is smaller than the allocated memory
 * is not resized
 *
 * \param sz size
 * \return true if the resize operation complete correctly
 *
 */

bool CudaMemory::resize(size_t sz)
{
	// if the allocated memory is enough, do not resize
	if (sz <= size())
		return true;

	//! Allocate the device memory if not done yet

	if (size() == 0)
		return allocate(sz);

	//! Create a new buffer, if sz is bigger than the actual size
	void * thm;

	//! Create a new buffer, if sz is bigger than the actual size
	void * tdm;

	if (dm != NULL)
	{
		if (this->sz < sz)
			CUDA_SAFE_CALL(cudaMalloc(&tdm,sz));

		//! copy from the old buffer to the new one

		CUDA_SAFE_CALL(cudaMemcpy(tdm,dm,size(),cudaMemcpyDeviceToDevice));
	}

	if (hm != NULL)
	{
		if (this->sz < sz)
			CUDA_SAFE_CALL(cudaHostAlloc(&thm,sz,cudaHostAllocMapped));

		//! copy from the old buffer to the new one

		CUDA_SAFE_CALL(cudaMemcpy(thm,hm,size(),cudaMemcpyHostToHost));
	}

	//! free the old buffer

	destroy();

	dm = tdm;
	hm = thm;

	//! change to the new buffer

	this->sz = sz;

	return true;
}

/*! \brief Return a readable pointer with your data
 *
 * Return a readable pointer with your data
 *
 */

void * CudaMemory::getPointer()
{
	//| allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	//! if the host buffer is synchronized with the device buffer return the host buffer

	if (is_hm_sync)
		return hm;

	//! copy from device to host memory

	CUDA_SAFE_CALL(cudaMemcpy(hm,dm,sz,cudaMemcpyDeviceToHost));

	return hm;
}
