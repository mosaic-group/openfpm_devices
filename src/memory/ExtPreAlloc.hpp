/* ExtPreAlloc.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Pietro Incardona
 */

#ifndef EXTPREALLOC_HPP_
#define EXTPREALLOC_HPP_

#include <stddef.h>
#include "memory.hpp"
#include <iostream>

/*! Preallocated memory sequence
 *
 * External pre-allocated memory, is a class that preallocate memory and than it answer
 * to a particular allocation pattern
 *
 * \warning zero sized allocation are removed from the request pattern
 *
 * \tparam Base memory allocation class [Example] HeapMemory or CudaMemory
 *
 *
 */

template<typename Mem>
class ExtPreAlloc : public memory
{
	//! Actual allocation pointer
	size_t a_seq ;

	//! Last allocation size
	size_t l_size;

	//! Main class for memory allocation
	Mem * mem;

	//! Reference counter
	long int ref_cnt;

public:

	virtual ~ExtPreAlloc()
	{
		if (ref_cnt != 0)
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	}

	//! Default constructor
	ExtPreAlloc()
	:a_seq(0),l_size(0),mem(NULL),ref_cnt(0)
	{
	}

	/*! \brief Preallocated memory sequence
	 *
	 * \param size number of bytes
	 * \param mem external memory, used if you want to keep the memory
	 *
	 */
	ExtPreAlloc(size_t size, Mem & mem)
	:a_seq(0),l_size(0),mem(&mem),ref_cnt(0)
	{
		// Allocate the total size of memory
		mem.resize(size);
	}


	/*! \brief Copy the memory from device to device
	 *
	 * \param m memory from where to copy
	 *
	 */
	bool copyDeviceToDevice(const ExtPreAlloc<Mem> & m)
	{
		return mem->copyDeviceToDevice(*m.mem);
	}

	/*! \brief special function to move memory from a raw device pointer
	 *
	 * \param start byte
	 * \param stop byte
	 *
	 * \param offset destination byte
	 *
	 */
	void deviceToDevice(void * ptr, size_t start, size_t stop, size_t offset)
	{
		mem->deviceToDevice(ptr,start,stop,offset);
	}

	constexpr static bool isDeviceHostSame()
	{
		return Mem::isDeviceHostSame();
	}

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

	//! flush the memory
	virtual bool flush() {return mem->flush();};

	/*! \brief fill host and device memory with the selected byte
	 *
	 *
	 */
	virtual void fill(unsigned char c)
	{
		mem->fill(c);
	}

	/*! \brief Allocate a chunk of memory
	 *
	 * Allocate a chunk of memory
	 *
	 * \param sz size of the chunk of memory to allocate in byte
	 *
	 */
	virtual bool allocate(size_t sz)
	{
		// Zero sized allocation are ignored
		if (sz == 0)
			return true;

		a_seq = l_size;
		l_size += sz;

		// Check we do not overflow the allocated memory
#ifdef SE_CLASS1

		if (l_size > mem->size())
			std::cerr << __FILE__ << ":" << __LINE__ << " Error requesting more memory than the allocated one" << std::endl;

#endif

		return true;
	}

	/*! \brief Allocate a chunk of memory
	 *
	 * Allocate a chunk of memory
	 *
	 * \param sz size of the chunk of memory to allocate in byte
	 *
	 */
	bool allocate_nocheck(size_t sz)
	{
		// Zero sized allocation are ignored
		if (sz == 0)
			return true;

		a_seq = l_size;
		l_size += sz;

		return true;
	}

	/*! \brief Return the end pointer of the previous allocated memory
	 *
	 * \return the pointer
	 *
	 */
	void * getPointerEnd()
	{
		return (char *)mem->getPointer() + l_size;
	}

	/*! \brief Return the device end pointer of the previous allocated memory
	 *
	 * \return the pointer
	 *
	 */
	void * getDevicePointerEnd()
	{
		return (char *)mem->getDevicePointer() + l_size;
	}

	/*! \brief The the base pointer of the preallocate memory
	 *
	 * \return the base pointer
	 *
	 */
	void * getPointerBase()
	{
		return mem->getPointer();
	}

	/*! \brief Return the pointer of the last allocation
	 *
	 * \return the pointer
	 *
	 */
	virtual void * getDevicePointer()
	{
		return (((unsigned char *)mem->getDevicePointer()) + a_seq );
	}

	/*! \brief Return the pointer of the last allocation
	 *
	 * \return the pointer
	 *
	 */
	virtual void hostToDevice()
	{
		mem->hostToDevice();
	}

	/*! \brief Return the pointer of the last allocation
	 *
	 * \return the pointer
	 *
	 */
	virtual void hostToDevice(size_t start, size_t stop)
	{
		mem->hostToDevice(start,stop);
	}

	//! Do nothing
	virtual void deviceToHost()
	{
		mem->deviceToHost();
	};

	//! Do nothing
	virtual void deviceToHost(size_t start, size_t stop)
	{
		mem->deviceToHost(start,stop);
	};

	/*! \brief Return the pointer of the last allocation
	 *
	 * \return the pointer
	 *
	 */
	virtual void * getPointer()
	{
		return (((unsigned char *)mem->getPointer()) + a_seq );
	}

	/*! \brief Return the pointer of the last allocation
	 *
	 * \return the pointer
	 *
	 */
	virtual const void * getPointer() const
	{
		return (((unsigned char *)mem->getPointer()) + a_seq);
	}

	/*! \brief Get the base memory pointer increased with an offset
	 *
	 * \param offset memory offset
	 *
	 */
	void * getPointerOffset(size_t offset)
	{
		return (((unsigned char *)mem->getPointer()) + offset);
	}

	/*! \brief Allocate or resize the allocated memory
	 *
	 * Resize the allocated memory, if request is smaller than the allocated, memory
	 * is not resized
	 *
	 * \param sz size
	 * \return true if the resize operation complete correctly
	 *
	 */

	virtual bool resize(size_t sz)
	{
		return allocate(sz);
	}

	/*! \brief Get the size of the LAST allocated memory
	 *
	 * Get the size of the allocated memory
	 *
	 * \return the size of the allocated memory
	 *
	 */

	virtual size_t size() const
	{
		return l_size;
	}

	/*! \brief Destroy memory
	 *
	 */

	void destroy()
	{
		mem->destroy();
	}

	/*! \brief Copy memory
	 *
	 */

	virtual bool copy(const memory & m)
	{
		return mem->copy(m);
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

	/*! \brief Calculate the total memory required to pack the message
	 *
	 * \return the total required memory
	 *
	 */
	static size_t calculateMem(std::vector<size_t> & mm)
	{
		size_t s = 0;

		for (size_t i = 0 ; i < mm.size() ; i++)
			s += mm[i];

		return s;
	}

	/*! \brief shift the pointer backward
	 *
	 * \warning when you shift backward the pointer, the last allocation is lost
	 * 			this mean that you have to do again an allocation.
	 *
	 * This function is useful to go ahead in memory and fill the memory later on
	 *
	 * \code

	  mem.allocate(16); <------ Here we allocate 16 byte but we do not fill it because
	  	  	  	  	  	  	    subsequently we do another allocation without using mem
	  unsigned char * start = (unsigned char *)mem.getPointer()
	  mem.allocate(100)

	  // ...
	  // ...
	  // Code that fill mem in some way and do other mem.allocate(...)
	  // ...
	  // ...

	  unsigned char * final = (unsigned char *)mem.getPointer()
	  mem.shift_backward(final - start);
	  mem.allocate(16);  <------ Here I am getting the same memory that I request for the
	  	  	  	  	  	  	  	 first allocate

	  // we now fill the memory

	  \endcode
	 *
	 *
	 *
	 * \param how many byte to shift
	 *
	 */
	void shift_backward(size_t sz)
	{
		a_seq -= sz;
		l_size = a_seq;
	}

	/*! \brief shift the pointer forward
	 *
	 * The same as shift backward, but in this case it move the pointer forward
	 *
	 * In general you use this function after the you went back with shift_backward
	 * and you have to move forward again
	 *
	 * \warning when you shift forward the pointer, the last allocation is lost
	 * 			this mean that you have to do again an allocation.
	 *
	 */
	void shift_forward(size_t sz)
	{
		a_seq += sz;
		l_size = a_seq;
	}

	/*! \brief Get offset
	 *
	 * \return the offset
	 *
	 */
	size_t getOffset()
	{
		return a_seq;
	}

	/*! \brief Get offset
	 *
	 * \return the offset
	 *
	 */
	size_t getOffsetEnd()
	{
		return l_size;
	}

	/*! \brief Reset the internal counters
	 *
	 *
	 */
	void reset()
	{
		a_seq = 0;
		l_size = 0;
	}
};

#endif /* PREALLOCHEAPMEMORY_HPP_ */
