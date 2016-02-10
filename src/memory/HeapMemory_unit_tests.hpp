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

	mem.allocate(FIRST_ALLOCATION);

	BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);

	// get the pointer of the allocated memory and fill

	unsigned char * ptr = (unsigned char *)mem.getPointer();
	for (size_t i = 0 ; i < mem.size() ; i++)
		ptr[i] = i;

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

	T dst = src;

	unsigned char * ptr2 = (unsigned char *)dst.getPointer();

	BOOST_REQUIRE(src.getPointer() != dst.getPointer());
	for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
	{
		unsigned char c=i;
		BOOST_REQUIRE_EQUAL(ptr2[i],c);
	}

	}
}

BOOST_AUTO_TEST_CASE( use )
{
	test<HeapMemory>();
#ifdef CUDA_GPU
	test<CudaMemory>();
#endif
}


BOOST_AUTO_TEST_SUITE_END()


#endif /* HEAPMEMORY_UNIT_TESTS_HPP_ */
