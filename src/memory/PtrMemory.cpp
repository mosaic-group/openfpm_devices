/*
 * PtrMemory.cpp
 *
 *  Created on: Apr 15, 2015
 *      Author: i-bird
 */

#ifndef PTRMEMORY_CPP_
#define PTRMEMORY_CPP_

#include "PtrMemory.hpp"
#include <cstddef>
#include <cstring>
#include <iostream>
#include <cstdint>

// If debugging mode include memory leak check
#ifdef SE_CLASS2
#include "Memleak_check.hpp"
#endif

/*! \brief Allocate a chunk of memory
 *
 * \param sz size of the chunk of memory to allocate in byte
 *
 */

bool PtrMemory::allocate(size_t sz)
{
	if (sz <= spm)
		return true;

	std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " allocation failed";
	return false;
}

/*! \brief destroy a chunk of memory
 *
 */
void PtrMemory::destroy()
{
}


/*! \brief copy the data from a pointer
 *
 *	\param ptr
 */
bool PtrMemory::copyFromPointer(const void * ptr,size_t sz)
{
	// memory copy

	memcpy(dm,ptr,sz);

	return true;
}

/*! \brief copy from device to device
 *
 * copy a piece of memory from device to device
 *
 * \param m PtrMemory from where to copy
 *
 * \return true if the memory is successful copy
 *
 */
bool PtrMemory::copyDeviceToDevice(const PtrMemory & m)
{
	//! The source buffer is too big to copy it

	if (m.spm > spm)
	{
		std::cerr << "Error " << __LINE__ << " " << __FILE__ << ": source buffer is too big to copy";
		return false;
	}

	// Copy the memory from m
	memcpy(dm,m.dm,m.spm);
	return true;
}

/*! \brief copy the memory
 *
 * \param m a memory interface
 *
 */
bool PtrMemory::copy(const memory & m)
{
	//! Here we try to cast memory into PtrMemory
	const PtrMemory * ofpm = dynamic_cast<const PtrMemory *>(&m);

	//! if we fail we get the pointer and simply copy from the pointer

	if (ofpm == NULL)
	{
		// copy the memory from device to host and from host to device

		return copyFromPointer(m.getPointer(),m.size());
	}
	else
	{
		// they are the same memory type, use cuda/thrust buffer copy

		return copyDeviceToDevice(*ofpm);
	}
	return false;
}

/*! \brief Get the size of the allocated memory
 *
 * Get the size of the allocated memory
 *
 * \return the size of the allocated memory
 *
 */

size_t PtrMemory::size() const
{
	return spm;
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

bool PtrMemory::resize(size_t sz)
{
	// if the allocated memory is enough, do not resize
	if (sz <= spm)
	{
		this->spm = sz;
		return true;
	}

	std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " allocation failed";
	return false;
}

/*! \brief Return a pointer to the allocated memory
 *
 * \return the pointer
 *
 */

void * PtrMemory::getPointer()
{
	return dm;
}

/*! \brief Return a pointer to the allocated memory
 *
 * \return the pointer
 *
 */

void * PtrMemory::getDevicePointer()
{
	return dm;
}

/*! \brief Return a pointer to the allocated memory
 *
 * \return the pointer
 *
 */

void * PtrMemory::getDevicePointerNoCopy()
{
	return dm;
}

/*! \brief Return a pointer to the allocated memory
 *
 * \return the pointer
 *
 */

const void * PtrMemory::getPointer() const
{
	return dm;
}


#endif /* PTRMEMORY_CPP_ */
