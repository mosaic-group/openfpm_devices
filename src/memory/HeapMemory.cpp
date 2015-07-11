/*
 * HeapMemory.cpp
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

#include "HeapMemory.hpp"
#include <cstddef>
#include <cstring>
#include <iostream>
#include <cstdint>

// If debugging mode include memory leak check
#ifdef MEMLEAK_CHECK
#include "Memleak_check.hpp"
#endif

/*! \brief Allocate a chunk of memory
 *
 * \param sz size of the chunk of memory to allocate in byte
 *
 */

bool HeapMemory::allocate(size_t sz)
{
	//! Allocate the device memory
	if (dm == NULL)
		dmOrig = new byte[sz+alignement];
	dm = dmOrig;

#ifdef MEMLEAK_CHECK
	check_new(dmOrig,sz+alignement);
#endif

	// align it, we do not know the size of the element we put 1
	// and we ignore the align check
	size_t sz_a = sz+alignement;
	align(alignement,1,(void *&)dm,sz_a);

	this->sz = sz;

	return true;
}

/*! \brief set the memory block to be aligned by this number
 *
 */
void HeapMemory::setAlignment(size_t align)
{
	this->alignement = align;
}

/*! \brief destroy a chunk of memory
 *
 * Destroy a chunk of memory
 *
 */
void HeapMemory::destroy()
{
	if (dmOrig != NULL)
		delete [] dmOrig;

#ifdef MEMLEAK_CHECK
	check_delete(dmOrig);
#endif
}


/*! \brief copy the data from a pointer
 *
 *
 *	\param ptr
 */
bool HeapMemory::copyFromPointer(void * ptr,size_t sz)
{
	// memory copy

	memcpy(dm,ptr,sz);

	return true;
}

/*! \brief copy from device to device
 *
 * copy a piece of memory from device to device
 *
 * \param CudaMemory from where to copy
 *
 */
bool HeapMemory::copyDeviceToDevice(HeapMemory & m)
{
	//! The source buffer is too big to copy it

	if (m.sz > sz)
	{
		std::cerr << "Error " << __LINE__ << __FILE__ << ": source buffer is too big to copy";
		return false;
	}

	// Copy the memory from m
	memcpy(dm,m.dm,m.sz);
	return true;
}

/*! \brief copy the memory
 *
 * \param m a memory interface
 *
 */
bool HeapMemory::copy(memory & m)
{
	//! Here we try to cast memory into HeapMemory
	HeapMemory * ofpm = dynamic_cast<HeapMemory *>(&m);

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

size_t HeapMemory::size()
{
	return sz;
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

bool HeapMemory::resize(size_t sz)
{
	// if the allocated memory is enough, do not resize
	if (sz <= size())
		return true;

	//! Allocate the device memory if not done yet

	if (size() == 0)
		return allocate(sz);

	//! Create a new buffer if sz is bigger than the actual size
	byte * tdm;
	byte * tdmOrig;
	tdmOrig = new byte[sz+alignement];
	tdm = tdmOrig;

	//! size plus alignment
	size_t sz_a = sz+alignement;

	this->sz = sz;

	//! align it
	align(alignement,1,(void *&)tdm,sz_a);

	//! copy from the old buffer to the new one

	memcpy(tdm,dm,size());

	//! free the old buffer

	destroy();

	//! change to the new buffer

	dm = tdm;
	dmOrig = tdmOrig;
	this->sz = sz;

	return true;
}

/*! \brief Return a readable pointer with your data
 *
 * Return a readable pointer with your data
 *
 */

void * HeapMemory::getPointer()
{
	return dm;
}
