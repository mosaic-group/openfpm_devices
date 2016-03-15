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
#ifdef SE_CLASS2
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
	else
		std::cerr << __FILE__ << ":" << __LINE__ << " error memory already allocated\n";

	dm = dmOrig;

#ifdef SE_CLASS2
	check_new(dmOrig,sz+alignement,HEAPMEMORY_EVENT,0);
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
#ifdef SE_CLASS2
	check_delete(dmOrig);
#endif

	if (dmOrig != NULL)
		delete [] dmOrig;
}


/*! \brief copy the data from a pointer
 *
 *
 *	\param ptr
 */
bool HeapMemory::copyFromPointer(const void * ptr,size_t sz)
{
	// memory copy

#ifdef SE_CLASS2
	check_valid(dm,sz);
	check_valid(ptr,sz);
#endif
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
bool HeapMemory::copyDeviceToDevice(const HeapMemory & m)
{
	//! The source buffer is too big to copy it

	if (m.sz > sz)
	{
		std::cerr << "Error " << __LINE__ << __FILE__ << ": source buffer is too big to copy";
		return false;
	}

#ifdef SE_CLASS2
	check_valid(dm,sz);
	check_valid(m.dm,sz);
#endif
	// Copy the memory from m
	memcpy(dm,m.dm,m.sz);
	return true;
}

/*! \brief copy the memory
 *
 * \param m a memory interface
 *
 */
bool HeapMemory::copy(const memory & m)
{
	//! Here we try to cast memory into HeapMemory
	const HeapMemory * ofpm = dynamic_cast<const HeapMemory *>(&m);

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

size_t HeapMemory::size() const
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
#ifdef SE_CLASS2
	check_new(tdmOrig,sz+alignement,HEAPMEMORY_EVENT,0);
#endif
	tdm = tdmOrig;

	//! size plus alignment
	size_t sz_a = sz+alignement;

	//! align it
	align(alignement,1,(void *&)tdm,sz_a);

	//! copy from the old buffer to the new one

#ifdef SE_CLASS2
	check_valid(tdm,size());
	check_valid(dm,size());
#endif
	memcpy(tdm,dm,size());

	this->sz = sz;

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

/*! \brief Return a readable pointer with your data
 *
 * Return a readable pointer with your data
 *
 */

const void * HeapMemory::getPointer() const
{
	return dm;
}
