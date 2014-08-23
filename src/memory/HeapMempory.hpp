/*
 * HeapMempory.hpp
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

/**
 * \brief This class create instructions to allocate, and destroy CPU memory
 *
 * This class create instructions to allocate, destroy, resize CPU buffer,
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

#include "memory.hpp"
#include <cstddef>

class HeapMemory : public memory
{
	//! Size of the memory
	size_t sz;

	//! device memory
	void * dm;

	//! allocate memory
	virtual bool allocate(size_t sz);
	//! destroy memory
	virtual void destroy();
	//! copy memory
	virtual bool copy(memory & m);
	//! the the size of the allocated memory
	virtual size_t size();
	//! resize the memory allocated
	virtual bool resize(size_t sz);
	//! get a readable pointer with the data
	void * getPointer()

	//! copy from same Heap to Heap
	void copyDeviceToDevice(HeapMemory & m);

	//! copy from Pointer to Heap
	void copyFromPointer(void * ptr);

	//! copy from a General device
	bool copy(memory & m);

	//! Constructor
	HeapMemory():dm(NULL),sz(0) {};

	~HeapMemory() {};
};

