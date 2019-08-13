/*
 * HeapMempory.hpp
 *
 *  Created on: Aug 17, 2014
 *      Author: Pietro Incardona
 */

#ifndef HEAP_MEMORY_HPP
#define HEAP_MEMORY_HPP

#include "config.h"
#include "memory.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "ShmAllocator_manager.hpp"

typedef unsigned char byte;

#define MEM_ALIGNMENT 32


/**
 * \brief This class allocate, and destroy CPU memory
 *
 *
 * ### Allocate memory
 *
 * \snippet HeapMemory_unit_tests.hpp Allocate some memory and fill with data
 *
 * ### Resize memory
 *
 * \snippet HeapMemory_unit_tests.hpp Resize the memory
 *
 * ### Shrink memory
 *
 * \snippet HeapMemory_unit_tests.hpp Shrink memory
 *
 */
class HeapMemory : public memory
{
	//! memory alignment
	size_t alignement;

	//! Size of the memory
	size_t sz;

	//! device memory
	byte * dm;
	//! original pointer (before alignment)
	byte * dmOrig;
	//! Reference counter
	long int ref_cnt;

	//! key for shared memory
	handle_shmem sh_handle;

	//! shared memory id
	int shmid = -1;

	//! copy from Pointer to Heap
	bool copyFromPointer(const void * ptr, size_t sz);

	//! Set alignment the memory will be aligned with this number
	void setAlignment(size_t align);

	// allocate memory internal
	byte * allocate_mem(size_t sz);

public:

	//! copy from same Heap to Heap
	bool copyDeviceToDevice(const HeapMemory & m);

	//! flush the memory
	virtual bool flush() {return true;};
	//! allocate memory
	virtual bool allocate(size_t sz);
	//! destroy memory
	virtual void destroy();
	//! copy memory
	virtual bool copy(const memory & m);
	//! the the size of the allocated memory
	virtual size_t size() const;
	//! resize the memory allocated
	virtual bool resize(size_t sz);
	//! get a readable pointer with the data
	virtual void * getPointer();

	//! get a readable pointer with the data
	virtual const void * getPointer() const;

	//! get a device pointer for HeapMemory getPointer and getDevicePointer are equivalents
	virtual void * getDevicePointer();

	/*! \brief fill host and device memory with the selected byte
	 *
	 *
	 */
	virtual void fill(unsigned char c);

	//! Do nothing
	virtual void hostToDevice(){};

	//! Do nothing
	virtual void deviceToHost(){};

	//! Do nothing
	virtual void deviceToHost(size_t start, size_t stop) {};

	//! Do nothing
	virtual void hostToDevice(size_t start, size_t stop) {};

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

	//! Return the shared memory key
	virtual handle_shmem get_shmem_handle()
	{
		return sh_handle;
	}

	//! Return the shared memory key
	virtual void set_shmem_handle(handle_shmem sh)
	{
		sh_handle = sh;
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

	// Copy the Heap memory
	HeapMemory & operator=(const HeapMemory & mem)
	{
		copy(mem);
		return *this;
	}

	// Copy the Heap memory
	HeapMemory(const HeapMemory & mem)
	:HeapMemory()
	{
		sh_handle.id = -1;
		allocate(mem.size());
		copy(mem);
	}

	HeapMemory(HeapMemory && mem) noexcept
	{
		sh_handle.id = -1;
		//! move
		alignement = mem.alignement;
		sz = mem.sz;
		dm = mem.dm;
		dmOrig = mem.dmOrig;
		ref_cnt = mem.ref_cnt;

		mem.alignement = MEM_ALIGNMENT;
		mem.sz = 0;
		mem.dm = NULL;
		mem.dmOrig = NULL;
		mem.ref_cnt = 0;
	}

	//! Constructor, we choose a default alignment of 32 for avx
	HeapMemory():alignement(MEM_ALIGNMENT),sz(0),dm(NULL),dmOrig(NULL),ref_cnt(0)
	{sh_handle.id = -1;};

	virtual ~HeapMemory() noexcept
	{
		if(ref_cnt == 0)
			HeapMemory::destroy();
		else
			std::cerr << "Error: " << __FILE__ << " " << __LINE__ << " destroying a live object" << "\n";
	};

	/*! \brief Swap the memory
	 *
	 * \param mem memory to swap
	 *
	 */
	void swap(HeapMemory & mem)
	{
		size_t alignement_tmp;
		size_t sz_tmp;
		byte * dm_tmp;
		byte * dmOrig_tmp;
		long int ref_cnt_tmp;

		alignement_tmp = alignement;
		sz_tmp = sz;
		dm_tmp = dm;
		dmOrig_tmp = dmOrig;
		ref_cnt_tmp = ref_cnt;

		alignement = mem.alignement;
		sz = mem.sz;
		dm = mem.dm;
		dmOrig = mem.dmOrig;
		ref_cnt = mem.ref_cnt;

		mem.alignement = alignement_tmp;
		mem.sz = sz_tmp;
		mem.dm = dm_tmp;
		mem.dmOrig = dmOrig_tmp;
		mem.ref_cnt = ref_cnt_tmp;
	}

	/*! \brief Return true if the device and the host pointer are the same
	 *
	 * \return true if they are the same
	 *
	 */
	static bool isDeviceHostSame()
	{
		return true;
	}
};


/*! \brief function to align a pointer equivalent to std::align
 *
 * function to align a pointer equivalent to std::align
 *
 */

inline void *align( std::size_t alignment, std::size_t size,
                    void *&ptr, std::size_t &space ) {
	std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
	std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
	std::size_t padding = aligned - pn;
	if ( space < size + padding ) return nullptr;
	space -= padding;
	return ptr = reinterpret_cast< void * >( aligned );
}

#endif
