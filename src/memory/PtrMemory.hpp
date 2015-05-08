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

#ifdef MEMLEAK_CHECK
#include "Memleak_check.hpp"
#endif

class PtrMemory : public memory
{
	//! Size of the pointed memory
	size_t spm;

	//! Pointed memory
	void * dm;

	//! Size of the memory
	size_t sz;

	//! Reference counter
	long int ref_cnt;

	//! copy from same Heap to Heap
	bool copyDeviceToDevice(PtrMemory & m);

	//! copy from Pointer to Heap
	bool copyFromPointer(void * ptr, size_t sz);

	//! Set alignment the memory will be aligned with this number
	void setAlignment(size_t align);

public:

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
	virtual void * getPointer();

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
	PtrMemory():spm(0),dm(NULL),sz(0),ref_cnt(0)
	{
#ifdef MEMLEAK_CHECK
		if (process_to_print == process_v_cl)
			std::cout << "Creating PtrMemory: " << this << "\n";
#endif
	};

	//! Constructor, we choose a default alignment of 32 for avx
	PtrMemory(void * ptr, size_t sz):spm(sz),dm(ptr),sz(0),ref_cnt(0)
	{
#ifdef MEMLEAK_CHECK
		if (process_to_print == process_v_cl)
			std::cout << "Creating PtrMemory: " << this << "\n";
#endif
	};

	~PtrMemory()
	{
#ifdef MEMLEAK_CHECK
		if (process_to_print == process_v_cl)
			std::cout << "Delete PtrMemory: " << this << "\n";
#endif

		if(ref_cnt == 0)
			destroy();
		else
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	};
};


#endif /* PTRMEMORY_HPP_ */
