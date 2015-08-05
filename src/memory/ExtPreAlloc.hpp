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
	// List of allowed allocation
	std::vector<size_t> sequence;
	// starting from 0 is the cumulative buffer of sequence
	// Example sequence   = 2,6,3,6
	//         sequence_c = 0,2,8,11
	std::vector<size_t> sequence_c;

	// Main class for memory allocation
	Mem * mem;
	//! Reference counter
	long int ref_cnt;

	ExtPreAlloc(const ExtPreAlloc & ext)
	:a_seq(0),mem(NULL),ref_cnt(0)
	{}

public:

	~ExtPreAlloc()
	{
		if (ref_cnt != 0)
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	}

	//! Default constructor
	ExtPreAlloc()
	:a_seq(0),mem(NULL),ref_cnt(0)
	{
	}

	/*! \brief Preallocated memory sequence
	 *
	 * \param sequence of allocation size
	 * \param mem external memory, used if you want to keep the memory
	 *
	 */
	ExtPreAlloc(const std::vector<size_t> & seq, Mem & mem)
	:a_seq(0),mem(&mem),ref_cnt(0)
	{
		size_t total_size = 0;

		// number of non zero
		size_t n_zero = 0;

		// remove zero size request
		for (size_t i = 0 ; i < seq.size(); i++)
		{
			if (seq[i] != 0)
				n_zero++;
		}

		// Resize the sequence
		sequence.resize(n_zero);
		sequence_c.resize(n_zero+1);
		size_t j = 0;
		for (size_t i = 0 ; i < seq.size() ; i++)
		{
			if (seq[i] != 0)
			{
				sequence[j] = seq[i];
				sequence_c[j] = total_size;
				total_size += seq[i];
				j++;
			}
		}
		sequence_c[j] = total_size;

		// Allocate the total size of memory
		mem.allocate(total_size);
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
		// Zero sized allocation are ignored
		if (sz == 0)
			return true;

		// Check that the size match

		if (sequence[a_seq] != sz)
		{
			std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << " expecting: " << sequence[a_seq] << " got: " << sz <<  " allocation failed \n";
			std::cerr << "NOTE: if you are using this structure with vector remember to use openfpm::vector<...>::calculateMem(...) to get the required allocation sequence\n";

			return false;
		}

		a_seq++;

		return true;
	}

	/*! \brief Return the pointer of the last allocation
	 *
	 * \return the pointer
	 *
	 */
	virtual void * getPointer()
	{
		if (a_seq == 0)
			return NULL;

		return (((unsigned char *)mem->getPointer()) + sequence_c[a_seq-1]);
	}

	/*! \brief Return the pointer you will get when you do the allocation ip
	 *
	 * This particular function exist because the allocation sequence is fixed a priori
	 *
	 * \param ip index of the pointer in the sequence
	 *
	 */
	void * getPointer(size_t ip)
	{
		return (((unsigned char *)mem->getPointer()) + sequence_c[ip]);
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
		mem->destroy();
	}

	/*! \brief Copy memory
	 *
	 */

	virtual bool copy(memory & m)
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
