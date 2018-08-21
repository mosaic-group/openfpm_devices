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
 * In the case of BHeapMemory if you try to shrink the memory nothing happen to the allocated memory.
 * To destroy the internal memory you must use destroy.
 *
 * BHeapMemory does not shrink the memory, but it shrink the buffer size. size() always return the buffer size
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
template<typename Memory>
class BMemory : public Memory
{
	//! size of the memory
	size_t buf_sz;

public:

	/*! \brief Copy the Heap memory
	 *
	 * \param mem memory to copy
	 *
	 */
	BMemory(const BMemory<Memory> & mem)
	:HeapMemory(mem),buf_sz(mem.size())
	{
	}

	/*! \brief Copy the Heap memory
	 *
	 * \param mem memory to copy
	 *
	 */
	BMemory(BMemory<Memory> && mem) noexcept
	:Memory((Memory &&)mem),buf_sz(mem.size())
	{
	}

	//! Constructor, we choose a default alignment of 32 for avx
	BMemory()
	:Memory(),buf_sz(0)
	{};

	//! Destructor
	virtual ~BMemory() noexcept
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
		bool ret = Memory::allocate(sz);

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
		bool ret = Memory::resize(sz);

		// if the allocated memory is enough, do not resize
		if (ret == true)
			buf_sz = sz;

		return ret;
	}

	/*! \brief Resize the buffer size
	 *
	 * \return the buffer size
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
		return Memory::size();
	}

	/*! \brief Copy the memory
	 *
	 * \param mem memory to copy
	 *
	 * \return itself
	 *
	 */
	BMemory & operator=(const BMemory<Memory> & mem)
	{
		buf_sz = mem.buf_sz;
		static_cast<Memory *>(this)->operator=(mem);

		return *this;
	}

	/*! \brief Copy the memory
	 *
	 * \param mem memory to copy
	 *
	 * \return itself
	 *
	 */
	BMemory & operator=(BMemory<Memory> && mem)
	{
		buf_sz = mem.buf_sz;
		static_cast<Memory *>(this)->operator=(mem);

		return *this;
	}

	/*! \brief Destroy the internal memory
	 *
	 *
	 */
	void destroy()
	{
		Memory::destroy();
		buf_sz = 0;
	}

	/*! \brief swap the two mwmory object
	 *
	 * \param mem Memory to swap with
	 *
	 */
	void swap(BMemory<Memory> & mem)
	{
		Memory::swap(mem);

		size_t buf_sz_t = mem.buf_sz;
		mem.buf_sz = buf_sz;
		buf_sz = buf_sz_t;
	}
};


#endif
