/* ExtPreAlloc.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Pietro Incardona
 */

#ifndef EXTPREALLOC_HPP_
#define EXTPREALLOC_HPP_

#include <stddef.h>

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
	// Actual allocation pointer
	size_t a_seq ;

	// Last allocation size
	size_t l_size;

	// Main class for memory allocation
	Mem * mem;
	//! Reference counter
	long int ref_cnt;

public:

	virtual ~ExtPreAlloc()
	{
		if (ref_cnt != 0)
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";

#ifdef SE_CLASS2
		// Eliminate all the old pointers

		// Eliminate all the old pointers (only if has been used)
		if (a_seq != 0)
		{
			for (size_t i = 0 ; i < sequence.size() ; i++)
				check_delete(getPointer(i));
		}

#endif
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

		// Check that the size match

		a_seq = l_size;
		l_size += sz;

#ifdef SE_CLASS2

		check_new(getPointer(),sz,HEAPMEMORY_EVENT,0);

#endif

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

#ifdef SE_CLASS2

		// Eliminate all the old pointers (only if has been used)
		if (a_seq != 0)
		{
			for (size_t i = 0 ; i < sequence.size() ; i++)
				check_delete(getPointer(i));
		}

#endif
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
};

#endif /* PREALLOCHEAPMEMORY_HPP_ */
