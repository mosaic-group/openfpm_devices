/*
 * HeapMempory.hpp
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

/**
 * \brief This class allocate, and destroy CPU memory
 *
 * Usage:
 *
 * HeapMemory m = new HeapMemory();
 *
 * m.allocate(1000*sizeof(int));
 * int * ptr = m.getPointer();
 * ptr[999] = 1000;
 * ....
 *
 *
 */

#ifndef HEAP_MEMORY_HPP
#define HEAP_MEMORY_HPP

#include "config.h"
#include "memory.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>

typedef unsigned char byte;

#define MEM_ALIGNMENT 32


class HeapMemory : public memory
{
	//! memory alignment
	size_t alignement;

	//! Size of the memory
	size_t sz;

	//! device memory
	byte * dm;
	//! original pointer (before alignment)
	byte * dmOrig;
	//! Reference counter
	long int ref_cnt;

	//! copy from same Heap to Heap
	bool copyDeviceToDevice(HeapMemory & m);

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

	//! Constructor, we choose a default alignment of 32 for avx
	HeapMemory():alignement(MEM_ALIGNMENT),sz(0),dm(NULL),dmOrig(NULL),ref_cnt(0) {};

	~HeapMemory()
	{
		if(ref_cnt == 0)
			destroy();
		else
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	};
};


/*! \brief function to align a pointer equivalent to std::align
 *
 * function to align a pointer equivalent to std::align
 *
 */

inline void *align( std::size_t alignment, std::size_t size,
                    void *&ptr, std::size_t &space ) {
	std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
	std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
	std::size_t padding = aligned - pn;
	if ( space < size + padding ) return nullptr;
	space -= padding;
	return ptr = reinterpret_cast< void * >( aligned );
}

#endif
