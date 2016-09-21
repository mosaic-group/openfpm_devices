/*
 * HeapMemory_unit_tests.hpp
 *
 *  Created on: Jul 9, 2015
 *      Author: i-bird
 */

#ifndef HEAPMEMORY_UNIT_TESTS_HPP_
#define HEAPMEMORY_UNIT_TESTS_HPP_

#include "config.h"

#include "memory/HeapMemory.hpp"
#include "memory/BHeapMemory.hpp"
#ifdef NVCC
#include "memory/CudaMemory.cuh"
#endif

BOOST_AUTO_TEST_SUITE( HeapMemory_test )

//! [Memory test constants]
#define FIRST_ALLOCATION 1024ul
#define SECOND_ALLOCATION 4096ul
//! [Memory test constants]

template<typename T> void test()
{
	//! [Allocate some memory and fill with data]
	T mem;

	BOOST_REQUIRE_EQUAL(mem.size(),0ul);

	mem.allocate(FIRST_ALLOCATION);

	BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);

	// get the pointer of the allocated memory and fill

	unsigned char * ptr = (unsigned char *)mem.getPointer();
	for (size_t i = 0 ; i < mem.size() ; i++)
		ptr[i] = i;

	mem.flush();

	//! [Allocate some memory and fill with data]

	//! [Resize the memory]
	mem.resize(SECOND_ALLOCATION);

	unsigned char * ptr2 = (unsigned char *)mem.getPointer();

	BOOST_REQUIRE_EQUAL(mem.size(),SECOND_ALLOCATION);
	BOOST_REQUIRE_EQUAL(mem.isInitialized(),false);

	//! [Resize the memory]

	// check that the data are retained
	for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
	{
		unsigned char c = i;
		BOOST_REQUIRE_EQUAL(ptr2[i],c);
	}

	//! [Shrink memory]

	mem.resize(1);
	BOOST_REQUIRE_EQUAL(mem.size(),SECOND_ALLOCATION);

	//! [Shrink memory]


	{
	//! [Copy memory]
	T src;
	T dst;

	src.allocate(FIRST_ALLOCATION);
	dst.allocate(SECOND_ALLOCATION);

	unsigned char * ptr = (unsigned char *)src.getPointer();
	for (size_t i = 0 ; i < src.size() ; i++)
		ptr[i] = i;

	dst.copy(src);

	for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
	{
		unsigned char c=i;
		BOOST_REQUIRE_EQUAL(ptr2[i],c);
	}

	//! [Copy Memory]
	}

	{
	T src;
	src.allocate(FIRST_ALLOCATION);

	unsigned char * ptr = (unsigned char *)src.getPointer();
	for (size_t i = 0 ; i < src.size() ; i++)
		ptr[i] = i;

	src.flush();

	T dst = src;

	unsigned char * ptr2 = (unsigned char *)dst.getPointer();

	BOOST_REQUIRE(src.getPointer() != dst.getPointer());
	for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
	{
		unsigned char c=i;
		BOOST_REQUIRE_EQUAL(ptr2[i],c);
	}

	mem.destroy();

	BOOST_REQUIRE_EQUAL(mem.size(),0ul);

	mem.allocate(FIRST_ALLOCATION);

	BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);

	}
}

template<typename T> void Btest()
{
	//! [BAllocate some memory and fill with data]
	T mem;

	mem.allocate(FIRST_ALLOCATION);

	BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);

	// get the pointer of the allocated memory and fill

	unsigned char * ptr = (unsigned char *)mem.getPointer();
	for (size_t i = 0 ; i < mem.size() ; i++)
		ptr[i] = i;

	//! [BAllocate some memory and fill with data]

	//! [BResize the memory]
	mem.resize(SECOND_ALLOCATION);

	unsigned char * ptr2 = (unsigned char *)mem.getPointer();

	BOOST_REQUIRE_EQUAL(mem.size(),SECOND_ALLOCATION);
	BOOST_REQUIRE_EQUAL(mem.isInitialized(),false);

	//! [BResize the memory]

	// check that the data are retained
	for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
	{
		unsigned char c = i;
		BOOST_REQUIRE_EQUAL(ptr2[i],c);
	}

	//! [BShrink memory]

	mem.resize(1);
	BOOST_REQUIRE_EQUAL(mem.size(),1ul);

	//! [BShrink memory]

	mem.destroy();

	BOOST_REQUIRE_EQUAL(mem.size(),0ul);

	mem.allocate(FIRST_ALLOCATION);

	BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);
}


template<typename T> void Stest()
{
	T mem1;
	T mem2;

	mem1.allocate(5*sizeof(size_t));
	mem2.allocate(6*sizeof(size_t));

	BOOST_REQUIRE_EQUAL(mem1.size(),5*sizeof(size_t));
	BOOST_REQUIRE_EQUAL(mem2.size(),6*sizeof(size_t));

	// get the pointer of the allocated memory and fill

	size_t * ptr1 = (size_t *)mem1.getPointer();
	size_t * ptr2 = (size_t *)mem2.getPointer();
	for (size_t i = 0 ; i < 5 ; i++)
		ptr1[i] = i;

	for (size_t i = 0 ; i < 6 ; i++)
		ptr2[i] = i+100;

	mem1.swap(mem2);

	bool ret = true;
	ptr1 = (size_t *)mem2.getPointer();
	ptr2 = (size_t *)mem1.getPointer();
	for (size_t i = 0 ; i < 5 ; i++)
		ret &= ptr1[i] == i;

	for (size_t i = 0 ; i < 6 ; i++)
		ret &= ptr2[i] == i+100;

	BOOST_REQUIRE_EQUAL(ret,true);

	BOOST_REQUIRE_EQUAL(mem1.size(),6*sizeof(size_t));
	BOOST_REQUIRE_EQUAL(mem2.size(),5*sizeof(size_t));
}

BOOST_AUTO_TEST_CASE( use_heap_memory )
{
	test<HeapMemory>();
#ifdef CUDA_GPU
	test<CudaMemory>();
#endif
}

BOOST_AUTO_TEST_CASE( use_cuda_memory )
{
	test<HeapMemory>();
#ifdef CUDA_GPU
	test<CudaMemory>();
#endif
}

BOOST_AUTO_TEST_CASE( use_bheap_memory )
{
	Btest<BHeapMemory>();
}

BOOST_AUTO_TEST_CASE( swap_heap_memory )
{
	Stest<HeapMemory>();
}

BOOST_AUTO_TEST_SUITE_END()


#endif /* HEAPMEMORY_UNIT_TESTS_HPP_ */
