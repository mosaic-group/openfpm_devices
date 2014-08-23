/*
 * HeapMemory.cpp
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

#include "HeapMemory.hpp"
#include <cstddef>

typedef unsigned char byte;

/*! \brief Allocate a chunk of memory
 *
 * Allocate a chunk of memory
 *
 * \param sz size of the chunk of memory to allocate in byte
 *
 */
bool HeapMemory::allocate(size_t sz)
{
	//! Allocate the device memory
	if (dm == NULL)
	dm = new byte[sz];

	this->sz = sz;

	return true;
}

/*! \brief destroy a chunk of memory
 *
 * Destroy a chunk of memory
 *
 */
void HeapMemory::destroy()
{
	delete [] dm;
}


/*! \brief copy the data from a pointer
 *
 * copy the data from a pointer
 *
 *	\param ptr
 */
void HeapMemory::copyFromPointer(void * ptr)
{
	// memory copy

	memcpy(ptr,dm,sz);
}

/*! \brief copy from device to device
 *
 * copy a piece of memory from device to device
 *
 * \param CudaMemory from where to copy
 *
 */
void HeapMemory::copyDeviceToDevice(HeapMemory & m)
{
	//! The source buffer is too big to copy it

	if (m.sz > sz)
	{
		std::cerr << "Error " << __LINE__ << __FILE__ << ": source buffer is too big to copy";
		return;
	}

	memcpy(m.dm,dm,m.sz);
}

/*! \brief copy from memory
 *
 * copy from memory
 *
 * \param m a memory interface
 *
 */
bool HeapMemory::copy(memory & m)
{
	//! Here we try to cast memory into OpenFPMwdeviceCudaMemory
	HeapMemory * ofpm = dynamic_cast<CudaMemory>(m);

	//! if we fail we get the pointer and simply copy from the pointer

	if (ofpm == NULL)
	{
		// copy the memory from device to host and from host to device

		return copyFromPointer(m.getPointer());
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
	//! Allocate the device memory if not done yet

	if (size() == 0)
		return allocate(sz);

	//! Create a new buffer if sz is bigger than the actual size
	void * tdm;
	tdm = new byte[sz];

	//! copy from the old buffer to the new one

	memcpy(dm,tdm,size());

	//! free the old buffer

	destroy();

	//! change to the new buffer

	dm = tdm;
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
