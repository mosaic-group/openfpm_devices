/*
 * CudaMemory.cu
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

/**
 * \brief This class create instructions to allocate, and destroy GPU memory
 * 
 * This class create instructions to allocate, destroy, resize GPU buffer, 
 * eventually if direct, comunication is not supported, it can instruction
 * to create an Host Pinned memory.
 * 
 * Usage:
 * 
 * CudaMemory m = new CudaMemory();
 * 
 * m.allocate();
 * int * ptr = m.getPointer();
 * *ptr[i] = 1000;
 * ....
 * 
 * 
 */

#ifndef CUDA_MEMORY_CUH_
#define CUDA_MEMORY_CUH_

#include "memory.hpp"

class CudaMemory : public memory
{
	//! Is the host memory synchronized with the GPU memory
	bool is_hm_sync;
	
	//! Size of the memory
	size_t sz;
	
	//! device memory
	void * dm;
	
	//! host memory
	void * hm;

	//! Allocate an host buffer
	void allocate_host(size_t sz);
	
	//! allocate memory
	virtual bool allocate(size_t sz);
	//! destroy memory
	virtual void destroy();
	//! copy from a General device
	virtual bool copy(memory & m);
	//! the the size of the allocated memory
	virtual size_t size();
	//! resize the momory allocated
	virtual bool resize(size_t sz);
	//! get a readable pointer with the data
	void * getPointer();
	
	//! copy from GPU to GPU buffer directly
	bool copyDeviceToDevice(CudaMemory & m);
	
	//! copy from Pointer to GPU
	bool copyFromPointer(void * ptr);
	
	//! This function notify that the device memory is not sync with
	//! the host memory, is called when a task is performed that write
	//! on the buffer
	void isNotSync() {is_hm_sync = false;}
	
	public:
	
	//! Constructor
	CudaMemory():is_hm_sync(false),sz(0),hm(0) {};
};

#endif

