#include "config.h"
#include <cstddef>
#include "CudaMemory.cuh"
#include "cuda_macro.h"
#include <cstring>

#define CUDA_EVENT 0x1201

/*! \brief Move the memory into device
 *
 * \return true if the memory is correctly flushed
 *
 */
bool CudaMemory::flush()
{
	if (hm != NULL && dm != NULL)
	{
		//! copy from host to device memory

		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaMemcpy(dm,hm,sz,cudaMemcpyHostToDevice));
		#else
		memcpy(dm,hm,sz);
		#endif
	}
	
	return true;
}

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
	{
		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaMalloc(&dm,sz));
		#else
		dm = new unsigned char[sz];
		#endif
	}
	else
	{
		if (sz != this->sz)
		{
			std::cout << __FILE__ << ":" << __LINE__ << " error FATAL: using allocate to resize the memory, please use resize." << std::endl;
			return false;
		}
	}

	this->sz = sz;

#ifdef FILL_CUDA_MEMORY_WITH_MINUS_ONE
	CUDA_SAFE_CALL(cudaMemset(dm,-1,sz))
#endif

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
		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaFree(dm));
		#else
		delete [] (unsigned char *)dm;
		#endif
		dm = NULL;
	}

	if (hm != NULL)
	{
		//! we invalidate hm
		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaFreeHost(hm));
		#else
		delete [] (unsigned char *)hm;
		#endif
		hm = NULL;
	}
	
	sz = 0;
}

/*! \brief copy memory from device to device
 *
 * \param external device pointer
 * \param start source starting point (where it start to copy)
 * \param stop end point
 * \param offset where to copy in the device pointer
 *
 */
void CudaMemory::deviceToDevice(void * ptr, size_t start, size_t stop, size_t offset)
{
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(((unsigned char *)dm)+offset,((unsigned char *)ptr)+start,(stop-start),cudaMemcpyDeviceToDevice));
	#else
	memcpy(((unsigned char *)dm)+offset,((unsigned char *)ptr)+start,(stop-start));
	#endif
}

/*! \brief Allocate the host buffer
 *
 * Allocate the host buffer
 *
 */
void CudaMemory::allocate_host(size_t sz) const
{
	if (hm == NULL)
	{
		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaHostAlloc(&hm,sz,cudaHostAllocMapped))
		#else
		hm = new unsigned char[sz];
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
bool CudaMemory::copyFromPointer(const void * ptr)
{
	// check if we have a host buffer, if not allocate it

	allocate_host(sz);

	// get the device pointer

	void * dvp;
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaHostGetDevicePointer(&dvp,hm,0));

	// memory copy

	memcpy(dvp,ptr,sz);
	#else
	memcpy(hm,ptr,sz);
	#endif

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
bool CudaMemory::copyDeviceToDevice(const CudaMemory & m)
{
	//! The source buffer is too big to copy it

	if (m.sz > sz)
	{
		std::cerr << "Error " << __LINE__ << __FILE__ << ": source buffer is too big to copy";
		return false;
	}

	//! Copy the memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(dm,m.dm,m.sz,cudaMemcpyDeviceToDevice));
	#else
	memcpy(dm,m.dm,m.sz);
	#endif

	return true;
}

/*! \brief copy from memory
 *
 * copy from memory
 *
 * \param m a memory interface
 *
 */
bool CudaMemory::copy(const memory & m)
{
	//! Here we try to cast memory into OpenFPMwdeviceCudaMemory
	const CudaMemory * ofpm = dynamic_cast<const CudaMemory *>(&m);

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

size_t CudaMemory::size() const
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
	if (sz <= CudaMemory::size())
	{return true;}

	//! Allocate the device memory if not done yet

	if (CudaMemory::size() == 0)
	{return allocate(sz);}

	//! Create a new buffer, if sz is bigger than the actual size
	void * thm = NULL;

	//! Create a new buffer, if sz is bigger than the actual size
	void * tdm = NULL;

	if (dm != NULL)
	{
		if (this->sz < sz)
		{
			#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
			CUDA_SAFE_CALL(cudaMalloc(&tdm,sz));
			#else
			tdm = new unsigned char [sz];
			#endif

#ifdef FILL_CUDA_MEMORY_WITH_MINUS_ONE
			#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
			CUDA_SAFE_CALL(cudaMemset(tdm,-1,sz));
			#else
			memset(tdm,-1,sz);
			#endif
#endif
		}

		//! copy from the old buffer to the new one
		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaMemcpy(tdm,dm,CudaMemory::size(),cudaMemcpyDeviceToDevice));
		#else
		memcpy(tdm,dm,CudaMemory::size());
		#endif
	}

	if (hm != NULL)
	{
		if (this->sz < sz)
		{
			#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
			CUDA_SAFE_CALL(cudaHostAlloc(&thm,sz,cudaHostAllocMapped));
			#else
			thm = new unsigned char [sz];
			#endif
		}

		//! copy from the old buffer to the new one
		#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
		CUDA_SAFE_CALL(cudaMemcpy(thm,hm,CudaMemory::size(),cudaMemcpyHostToHost));
		#else
		memcpy(thm,hm,CudaMemory::size());
		#endif
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
 * \return a readable pointer with your data
 *
 */

void * CudaMemory::getPointer()
{
	// allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	return hm;
}

/*! \brief Return a readable pointer with your data
 *
 * \return a readable pointer with your data
 *
 */

void CudaMemory::deviceToHost()
{
	// allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	//! copy from device to host memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(hm,dm,sz,cudaMemcpyDeviceToHost));
	#else
	memcpy(hm,dm,sz);
	#endif
}

/*! \brief It transfer to device memory from the host of another memory
 *
 * \param mem the other memory object
 *
 */
void CudaMemory::deviceToHost(CudaMemory & mem)
{
	// allocate an host memory if not allocated
	if (mem.hm == NULL)
		mem.allocate_host(sz);

	if (mem.sz > sz)
	{resize(mem.sz);}

	//! copy from device to host memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(mem.hm,dm,mem.sz,cudaMemcpyDeviceToHost));
	#else
	memcpy(mem.hm,dm,mem.sz);
	#endif
}

/*! \brief It transfer to device memory from the host of another memory
 *
 * \param mem the other memory object
 *
 */
void CudaMemory::hostToDevice(CudaMemory & mem)
{
	// allocate an host memory if not allocated
	if (mem.hm == NULL)
		mem.allocate_host(sz);

	if (mem.sz > sz)
	{resize(mem.sz);}

	//! copy from device to host memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(dm,mem.hm,mem.sz,cudaMemcpyHostToDevice));
	#else
	memcpy(dm,mem.hm,mem.sz);
	#endif
}

void CudaMemory::hostToDevice(size_t start, size_t stop)
{
	// allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	//! copy from device to host memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(((unsigned char *)dm)+start,((unsigned char *)hm)+start,(stop-start),cudaMemcpyHostToDevice));
	#else
	memcpy(((unsigned char *)dm)+start,((unsigned char *)hm)+start,(stop-start));
	#endif
}

/*! \brief Return a readable pointer with your data
 *
 * \return a readable pointer with your data
 *
 */
void CudaMemory::deviceToHost(size_t start, size_t stop)
{
	// allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	//! copy from device to host memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(((unsigned char *)hm)+start,((unsigned char *)dm)+start,(stop-start),cudaMemcpyDeviceToHost));
	#else
	memcpy(((unsigned char *)hm)+start,((unsigned char *)dm)+start,(stop-start));
	#endif
}



/*! \brief Return a readable pointer with your data
 *
 * \return a readable pointer with your data
 *
 */

const void * CudaMemory::getPointer() const
{
	// allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	return hm;
}

/*! \brief fill host and device memory with the selected byte
 *
 *
 */
void CudaMemory::fill(unsigned char c)
{
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemset(dm,c,size()));
	#else
	memset(dm,c,size());
	#endif
	if (hm != NULL)
	{memset(hm,c,size());}
}

/*! \brief Return the CUDA device pointer
 *
 * \return CUDA device pointer
 *
 */
void * CudaMemory::getDevicePointer()
{
	return dm;
}

/*! \brief Return a readable pointer with your data
 *
 * \return a readable pointer with your data
 *
 */

void CudaMemory::hostToDevice()
{
	// allocate an host memory if not allocated
	if (hm == NULL)
		allocate_host(sz);

	//! copy from device to host memory
	#if defined(CUDA_GPU) && !defined(CUDA_ON_CPU)
	CUDA_SAFE_CALL(cudaMemcpy(dm,hm,sz,cudaMemcpyHostToDevice));
	#else
	memcpy(dm,hm,sz);
	#endif
}


/*! \brief Swap the memory
 *
 * \param mem memory to swap
 *
 */
void CudaMemory::swap(CudaMemory & mem)
{
	size_t sz_tmp;
	void * dm_tmp;
//	long int ref_cnt_tmp;
	bool is_hm_sync_tmp;
	void * hm_tmp;

	hm_tmp = hm;
	is_hm_sync_tmp = is_hm_sync;
	sz_tmp = sz;
	dm_tmp = dm;
//	ref_cnt_tmp = ref_cnt;

	hm = mem.hm;
	is_hm_sync = mem.is_hm_sync;
	sz = mem.sz;
	dm = mem.dm;
	ref_cnt = mem.ref_cnt;

	mem.hm = hm_tmp;
	mem.is_hm_sync = is_hm_sync_tmp;
	mem.sz = sz_tmp;
	mem.dm = dm_tmp;
//	mem.ref_cnt = ref_cnt_tmp;
}
