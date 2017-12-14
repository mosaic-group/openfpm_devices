/*
 * PtrMemory.hpp
 *
 *  Created on: Apr 15, 2015
 *      Author: Pietro Incardona
 */

#ifndef PTRMEMORY_HPP_
#define PTRMEMORY_HPP_


/**
 * \brief This class give memory from a preallocated memory, memory destruction is not performed
 *
 * Useful to shape pieces of memory
 *
 * Usage:
 *
 * void * ptr = new int[1000]
 *
 * PtrMemory m = new PtrMemory(ptr,1000);
 *
 * m.allocate();
 * int * ptr = m.getPointer();
 * *ptr[999] = 1000;
 * ....
 *
 * delete[] ptr;
 *
 */

#include "config.h"
#include "memory.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>

#ifdef SE_CLASS2
#include "Memleak_check.hpp"
#endif

class PtrMemory : public memory
{
	//! Size of the pointed memory
	size_t spm;

	//! Pointed memory
	void * dm;

	//! Reference counter
	long int ref_cnt;

	//! copy from same Heap to Heap
	bool copyDeviceToDevice(const PtrMemory & m);

	//! copy from Pointer to Heap
	bool copyFromPointer(const void * ptr, size_t sz);

	//! Set alignment the memory will be aligned with this number
	void setAlignment(size_t align);

public:

	//! flush the memory
	virtual bool flush() {return true;};
	//! allocate memory
	virtual bool allocate(size_t sz);
	//! destroy memory
	virtual void destroy();
	//! copy memory
	virtual bool copy(const memory & m);
	//! the the size of the allocated memory
	virtual size_t size() const;
	//! resize the memory allocated
	virtual bool resize(size_t sz);
	//! get a readable pointer with the data
	virtual void * getPointer();

	//! get a readable pointer with the data
	virtual const void * getPointer() const;

	//! get a readable pointer with the data
	virtual void * getDevicePointer();

	//! Do nothing
	virtual void deviceToHost(){};

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

	/*! \brief Allocated Memory is already initialized
	 *
	 * \return true
	 *
	 */
	bool isInitialized()
	{
		return true;
	}

	// Default constructor
	PtrMemory():spm(0),dm(NULL),ref_cnt(0)
	{
	};

	//! Constructor, we choose a default alignment of 32 for avx
	PtrMemory(void * ptr, size_t sz):spm(sz),dm(ptr),ref_cnt(0)
	{
	};

	~PtrMemory()
	{
		if(ref_cnt == 0)
			destroy();
		else
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	};
};


#endif /* PTRMEMORY_HPP_ */
