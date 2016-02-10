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

#include "config.h"
#include "memory.hpp"
#include <iostream>
#ifdef SE_CLASS2
#include "Memleak_check.hpp"
#endif

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

	//! Reference counter
	size_t ref_cnt;
	
	//! Allocate an host buffer
	void allocate_host(size_t sz);
	
public:
	
	//! allocate memory
	virtual bool allocate(size_t sz);
	//! destroy memory
	virtual void destroy();
	//! copy from a General device
	virtual bool copy(const memory & m);
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
	
	//! Constructor
	CudaMemory():is_hm_sync(true),sz(0),dm(0),hm(0),ref_cnt(0) {};
	
	//! Destructor
	~CudaMemory()	
	{
		if(ref_cnt == 0)
			destroy();
		else
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n"; 
	};
};

#endif

