/*
 * PreAllocHeapMemory.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: i-bird
 */

#ifndef PREALLOCHEAPMEMORY_HPP_
#define PREALLOCHEAPMEMORY_HPP_

#include "HeapMemory.hpp"

/*! Preallocated memory sequence
 *
 * It is an allocator that respond to some allocation sequence
 *
 * \tparam number of allocation in the sequence
 *
 */

template<unsigned int N>
class PreAllocHeapMemory : public memory
{
	// Actual allocation pointer
	size_t a_seq ;
	// List of allowed allocation
	size_t sequence[N];
	// starting from 0 is the cumulative buffer of sequence
	// Example sequence   = 2,6,3,6
	//         sequence_c = 0,2,8,11
	size_t sequence_c[N];

	// Main class for memory allocation
	HeapMemory hp;
	//! Reference counter
	long int ref_cnt;


public:

	~PreAllocHeapMemory()
	{
		if (ref_cnt != 0)
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	}

	//! Default constructor
	PreAllocHeapMemory()
	:a_seq(0),ref_cnt(0)
	{}

	/*! \brief Preallocated memory sequence
	 *
	 * \param sequence of allocation size
	 *
	 */
	PreAllocHeapMemory(size_t (& sequence)[N])
	:a_seq(0),ref_cnt(0)
	{
		size_t total_size = 0;

		for (size_t i = 0 ; i < N ; i++)
		{
			this->sequence[i] = sequence[i];
			this->sequence_c[i] = total_size;
			total_size += sequence[i];
		}

		// Allocate the total size of memory

		hp.allocate(total_size);
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

	/*! \brief Allocate a chunk of memory
	 *
	 * Allocate a chunk of memory
	 *
	 * \param sz size of the chunk of memory to allocate in byte
	 *
	 */
	virtual bool allocate(size_t sz)
	{
		// Check that the size match

		if (sequence[a_seq] != sz)
		{
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " allocation failed";
			return false;
		}

		a_seq++;

		return true;
	}

	/*! \brief Return a readable pointer with your data
	 *
	 * Return a readable pointer with your data
	 *
	 */
	virtual void * getPointer()
	{
		if (a_seq == 0)
			return NULL;

		return (((unsigned char *)hp.getPointer()) +  sequence_c[a_seq-1]);
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

	/*! \brief Get the size of the allocated memory
	 *
	 * Get the size of the allocated memory
	 *
	 * \return the size of the allocated memory
	 *
	 */

	virtual size_t size()
	{
		if (a_seq == 0)
			return 0;

		return sequence[a_seq-1];
	}

	/*! \brief Destroy memory
	 *
	 */

	void destroy()
	{
		hp.destroy();
	}

	/*! \brief Copy memory
	 *
	 */

	virtual bool copy(memory & m)
	{
		return hp.copy(m);
	}
};

#endif /* PREALLOCHEAPMEMORY_HPP_ */
