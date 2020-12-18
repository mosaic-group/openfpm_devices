/*
 * CudaMemory.cu
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

/**
 * \brief This class create instructions to allocate, and destroy GPU memory
 * 
 * This class allocate, destroy, resize GPU buffer, 
 * eventually if direct, comunication is not supported, it can instruction
 * to create an Host Pinned memory.
 * 
 * Usage:
 * 
 * CudaMemory m = new CudaMemory();
 * 
 * m.allocate(1000*sizeof(int));
 * int * ptr = m.getPointer();
 * ptr[999] = 1000;
 * ....
 * 
 * 
 */

#ifndef CUDA_MEMORY_CUH_
#define CUDA_MEMORY_CUH_

#if __CUDACC_VER_MAJOR__ < 9
#define EXCEPT_MC
#else
#define EXCEPT_MC noexcept
#endif

#include "config.h"
#include "memory.hpp"
#include <iostream>

#if defined(__NVCC__)  && !defined(CUDA_ON_CPU)
#include <cuda_runtime.h>
#else
#include "util/cuda_util.hpp"
#endif

extern size_t TotCudaMemoryAllocated;

/*! \brief given an alignment and an alignment it return the smallest number numiple of the alignment
 *         such that the value returned is bigger ot equal that the number given
 *
 *         alignment 8 number 2 it return 8
 *         alignment 8 number 9 it return 16
 *
 * \param al alignment
 * \param number
 *
 */
__device__ inline size_t align_number_device(size_t al, size_t number)
{
	return number + ((number % al) != 0)*(al - number % al);
}

//! Is an array to report general error can happen in CUDA
static __device__ unsigned char global_cuda_error_array[256];

class CudaMemory : public memory
{
	//! Is the host memory synchronized with the GPU memory
	bool is_hm_sync;
	
	//! Size of the memory
	size_t sz;
	
	//! device memory
	void * dm;
	
	//! host memory
	mutable void * hm;

	//! Reference counter
	size_t ref_cnt;
	
	//! Allocate an host buffer
	void allocate_host(size_t sz) const;
	
	//! copy from Pointer to GPU
	bool copyFromPointer(const void * ptr);
	
public:
	
	//! copy from GPU to GPU buffer directly
	bool copyDeviceToDevice(const CudaMemory & m);

	//! flush the memory
	virtual bool flush();
	//! allocate memory
	virtual bool allocate(size_t sz);
	//! destroy memory
	virtual void destroy();
	//! copy from a General device
	virtual bool copy(const memory & m);
	//! the the size of the allocated memory
	virtual size_t size() const;
	//! resize the momory allocated
	virtual bool resize(size_t sz);
	//! get a readable pointer with the data
	virtual void * getPointer();
	
	//! get a readable pointer with the data
	virtual const void * getPointer() const;
	
	//! get a readable pointer with the data
	virtual void * getDevicePointer();

	//! Move memory from host to device
	virtual void hostToDevice();

	//! Move memory from device to host
	virtual void deviceToHost();

	//! Move memory from device to host, just the selected chunk
	virtual void deviceToHost(size_t start, size_t stop);

	//! Move memory from host to device, just the selected chunk
	virtual void hostToDevice(size_t start, size_t top);

	//! host to device using external memory (this host memory is copied into mem device memory)
	void hostToDevice(CudaMemory & mem);

	//! device to host using external memory (this device memory is copied into mem host memory)
	void deviceToHost(CudaMemory & mem);

	//! fill the buffer with a byte
	virtual void fill(unsigned char c);

	//! This function notify that the device memory is not sync with
	//! the host memory, is called when a task is performed that write
	//! on the buffer
	void isNotSync() {is_hm_sync = false;}
	
	public:
	
	//! Increment the reference counter
	virtual void incRef()
	{ref_cnt++;}

	//! Decrement the reference counter
	virtual void decRef()
	{ref_cnt--;}
	
	//! Return the reference counter
	virtual long int ref()
	{
		return ref_cnt;
	}

	/*! \brief Allocated Memory is never initialized
	 *
	 * \return false
	 *
	 */
	bool isInitialized()
	{
		return false;
	}
	
	// Copy the memory (device and host)
	CudaMemory & operator=(const CudaMemory & mem)
	{
		copy(mem);
		return *this;
	}

	// Copy the Cuda memory
	CudaMemory(const CudaMemory & mem)
	:CudaMemory()
	{
		allocate(mem.size());
		copy(mem);
	}

	CudaMemory(CudaMemory && mem) EXCEPT_MC
	{
		is_hm_sync = mem.is_hm_sync;
		sz = mem.sz;
		dm = mem.dm;
		hm = mem.hm;
		ref_cnt = mem.ref_cnt;

		// reset mem
		mem.is_hm_sync = false;
		mem.sz = 0;
		mem.dm = NULL;
		mem.hm = NULL;
		mem.ref_cnt = 0;
	}
	
	//! Constructor
	CudaMemory():is_hm_sync(true),sz(0),dm(0),hm(0),ref_cnt(0) {};
	
	//! Constructor
	CudaMemory(size_t sz):is_hm_sync(true),sz(0),dm(0),hm(0),ref_cnt(0)
	{
		allocate(sz);
	};

	//! Destructor
	~CudaMemory()	
	{
		if(ref_cnt == 0)
			destroy();
		else
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n"; 
	};

	/*! \brief copy memory from device to device
	 *
	 * \param external device pointer
	 * \param start source starting point (where it start to copy)
	 * \param stop end point
	 * \param offset where to copy in the device pointer
	 *
	 */
	void deviceToDevice(void * ptr, size_t start, size_t stop, size_t offset);

	void swap(CudaMemory & mem);

	/*! \brief Return true if the device and the host pointer are the same
	 *
	 * \return true if they are the same
	 *
	 */
	static bool isDeviceHostSame()
	{
		return false;
	}

	/*! \brief return the device memory
	 *
	 * \see equivalent to getDevicePointer()
	 *
	 */
	void * toKernel()
	{
		return getDevicePointer();
	}
};


#endif

