/*
 * RHeapMempory.hpp
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */


#ifndef BHEAP_MEMORY_HPP
#define BHEAP_MEMORY_HPP

#include "config.h"
#include "memory.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>

typedef unsigned char byte;

#define MEM_ALIGNMENT 32

/**
 * \brief It is like HeapMemory but buffered
 *
 * The concept of buffer is different from allocated memory. The buffer size can be <= the allocated memory
 *
 * It differs from HeapMemory in resize behavior.
 *
 * In the case of HeapMemory if you try to shrink the memory nothing happen to the allocated memory.
 * To destroy the internal memory you must use destroy.
 *
 * BHeapMemory does not shrink the memory, but it shrink the buffer size. size will always return the buffer size
 *
 * ### Allocate memory
 *
 * \snippet HeapMemory_unit_tests.hpp BAllocate some memory and fill with data
 *
 * ### Resize memory
 *
 * \snippet HeapMemory_unit_tests.hpp BResize the memory
 *
 * ### Shrink memory
 *
 * \snippet HeapMemory_unit_tests.hpp BShrink memory
 *
 */
class BHeapMemory : public HeapMemory
{
	size_t buf_sz;

public:

	// Copy the Heap memory
	BHeapMemory(const BHeapMemory & mem)
	:HeapMemory(mem)
	{
	}

	BHeapMemory(BHeapMemory && mem) noexcept
	:HeapMemory((HeapMemory &&)mem)
	{
		buf_sz = mem.buf_sz;
	}

	//! Constructor, we choose a default alignment of 32 for avx
	BHeapMemory()
	:HeapMemory(),buf_sz(0)
	{};

	virtual ~BHeapMemory()
	{
	};

	/*! \brief allocate the memory
	 *
	 * Resize the allocated memory, if request is smaller than the allocated memory
	 * is not resized
	 *
	 * \param sz size
	 * \return true if the resize operation complete correctly
	 *
	 */
	virtual bool allocate(size_t sz)
	{
		bool ret = HeapMemory::allocate(sz);

		if (ret == true)
			buf_sz = sz;

		return ret;
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
	virtual bool resize(size_t sz)
	{
		bool ret = HeapMemory::resize(sz);

		// if the allocated memory is enough, do not resize
		if (ret == true)
			buf_sz = sz;

		return ret;
	}

	/*! \brief Resize the buffer size
	 *
	 * Resize the buffer size,
	 *
	 * \param sz size
	 * \return true if the resize operation complete correctly
	 *
	 */
	virtual size_t size() const
	{
		return buf_sz;
	}

	/*! \brief Return the memory size
	 *
	 *
	 * \return The allocated memory size
	 *
	 */
	size_t msize()
	{
		return HeapMemory::size();
	}

	/*! \brief Copy the memory
	 *
	 *
	 */
	BHeapMemory & operator=(const BHeapMemory & mem)
	{
		buf_sz = mem.buf_sz;
		static_cast<HeapMemory *>(this)->operator=(mem);

		return *this;
	}

	/*! \brief Copy the memory
	 *
	 *
	 */
	BHeapMemory & operator=(BHeapMemory && mem)
	{
		buf_sz = mem.buf_sz;
		static_cast<HeapMemory *>(this)->operator=(mem);

		return *this;
	}

	/*! \brief Destroy the internal memory
	 *
	 *
	 */
	void destroy()
	{
		HeapMemory::destroy();
		buf_sz = 0;
	}
};


#endif
