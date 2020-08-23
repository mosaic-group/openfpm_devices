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
#include "memory/ShmBuffer.hpp"
#include "timer.hpp"
#ifdef CUDA_GPU
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
	{ptr[i] = i;}

	mem.hostToDevice();

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
	{ptr[i] = i;}

	src.hostToDevice();
	dst.copy(src);
	dst.deviceToHost();

    unsigned char * ptr2 = (unsigned char *)dst.getPointer();
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
	{ptr[i] = i;}

	src.hostToDevice();
	T dst = src;
	dst.deviceToHost();

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

template<typename T> void test_shm()
{
    //! [Allocate some memory and fill with data]
    T mem;
    handle_shmem h = create_shmanager().create("/", 1);
    mem.set_shmem_handle(h);

    BOOST_REQUIRE_EQUAL(mem.size(),0ul);

    mem.allocate(FIRST_ALLOCATION);

    ShmBuffer *memBuf;

    memBuf = new ShmBuffer("/", 1, false);
    ptr_with_size pws{};

    bool finish = false;
    size_t bufSize = 0;

    std::thread th1([memBuf, &pws, &finish, &bufSize]()
    {
        while(!finish) {
            memBuf->update_key(true);
            pws = memBuf->attach();
            bufSize = pws.size - extra_pad - MEM_ALIGNMENT;
            memBuf->detach(false);
        }
    });

    timer t;
    t.start();
    while(bufSize != FIRST_ALLOCATION)
    {
        if(t.getwct()>2.0)
        {
            break;
        }
    }

    BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);
    BOOST_REQUIRE_EQUAL(mem.size(),bufSize);

    std::cout<<"Checked the first allocation"<<std::endl;

    // get the pointer of the allocated memory and fill

    unsigned char * ptr = (unsigned char *)mem.getPointer();
    for (size_t i = 0 ; i < mem.size() ; i++)
    {ptr[i] = i;}

    mem.hostToDevice();

    //! [Allocate some memory and fill with data]

    //! [Resize the memory]
    mem.resize(SECOND_ALLOCATION);

//    memBuf->update_key(true);
//    std::cout<<"Updated key"<<std::endl;
//    pws = memBuf->attach();
//    std::cout<<"Attached"<<std::endl;

    t.start();
    while(bufSize != SECOND_ALLOCATION)
    {
        if(t.getwct()>2.0)
        {
            break;
        }
    }

    unsigned char * ptr2 = (unsigned char *)mem.getPointer();

    BOOST_REQUIRE_EQUAL(mem.size(),SECOND_ALLOCATION);
    BOOST_REQUIRE_EQUAL(mem.size(),bufSize);
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

    sleep(2);

    BOOST_REQUIRE_EQUAL(mem.size(),SECOND_ALLOCATION);
    BOOST_REQUIRE_EQUAL(mem.size(),bufSize);
    std::cout<<"Passed first shrink mem test"<<std::endl;

    //! [Shrink memory]


    {
        //! [Copy memory]

        handle_shmem h2 = create_shmanager().create("/", 2);
        handle_shmem h3 = create_shmanager().create("/", 3);

        T src;
        src.set_shmem_handle(h2);
        T dst;
        dst.set_shmem_handle(h3);

        src.allocate(FIRST_ALLOCATION);
        dst.allocate(SECOND_ALLOCATION);

        unsigned char * ptr = (unsigned char *)src.getPointer();
        for (size_t i = 0 ; i < src.size() ; i++)
        {ptr[i] = i;}

        src.hostToDevice();
        dst.copy(src);
        dst.deviceToHost();

        unsigned char * ptr2 = (unsigned char *)dst.getPointer();
        for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
        {
            unsigned char c=i;
            BOOST_REQUIRE_EQUAL(ptr2[i],c);
        }

        ShmBuffer *buf2;
        buf2 = new ShmBuffer("/", 2, false);
        ptr_with_size pws2{};

        buf2->update_key(true);
        pws2 = buf2->attach();

        ShmBuffer *buf3;
        buf3 = new ShmBuffer("/", 3, false);
        ptr_with_size pws3{};

        buf3->update_key(true);
        pws3 = buf3->attach();

        BOOST_REQUIRE_EQUAL(pws2.size - MEM_ALIGNMENT - extra_pad, FIRST_ALLOCATION);
        BOOST_REQUIRE_EQUAL(pws3.size - MEM_ALIGNMENT - extra_pad, SECOND_ALLOCATION);

        //! [Copy Memory]
    }

    {
        handle_shmem h4 = create_shmanager().create("/", 4);

        T src;
        src.set_shmem_handle(h4);
        src.allocate(FIRST_ALLOCATION);

        unsigned char * ptr = (unsigned char *)src.getPointer();
        for (size_t i = 0 ; i < src.size() ; i++)
        {ptr[i] = i;}

        src.hostToDevice();
        T dst = src;
        dst.deviceToHost();

        unsigned char * ptr2 = (unsigned char *)dst.getPointer();

        for (size_t i = 0 ; i < FIRST_ALLOCATION ; i++)
        {
            unsigned char c=i;
            BOOST_REQUIRE_EQUAL(ptr2[i],c);
        }

        mem.destroy();

        int tok1 = PROJ_ID(1, 0);
        int tok2  =PROJ_ID(1, 1);

        int key1 = ftok("/", tok1);
        int key2 = ftok("/", tok2);

        int shmid1;
        int shmid2;
        int errno1 = 0;
        int errno2 = 0;

        t.start();
        while(errno1 != ENOENT && errno2 != ENOENT)
        {
            shmid1 = shmget(key1, 0, 0);
            errno1 = errno;
            shmid2 = shmget(key2, 0, 0);
            errno2 = errno;
            if(t.getwct()>2.0)
            {
                break;
            }
        }

        BOOST_REQUIRE_EQUAL(mem.size(),0ul);
        BOOST_REQUIRE_EQUAL(errno1,ENOENT);
        BOOST_REQUIRE_EQUAL(errno2,ENOENT);

        mem.allocate(FIRST_ALLOCATION);

        t.start();
        while(bufSize != FIRST_ALLOCATION)
        {
            if(t.getwct()>2.0)
            {
                break;
            }
        }
        BOOST_REQUIRE_EQUAL(mem.size(),FIRST_ALLOCATION);
        BOOST_REQUIRE_EQUAL(bufSize,FIRST_ALLOCATION);
        finish = true;
        th1.join();
    }
}

void shm_usage()
{
    //! [Create a shared memory unit and fill it with data]
    handle_shmem h = create_shmanager().create("/", 0);
    unsigned char * ptr = (unsigned char*)create_shmanager().alloc(h, FIRST_ALLOCATION);

    for(size_t i = 0; i < FIRST_ALLOCATION; i++)
    {
        ptr[i] = i;
    }

    //! [Create a reader and read memory]

    ShmBuffer * buf = new ShmBuffer("/", 0, false);
    ptr_with_size pws{};

    buf->update_key(true);
    pws = buf->attach();
    buf->detach(false); // Detach from previous memory. Currently, no previous memory, so will do nothing

    BOOST_REQUIRE_EQUAL(pws.size, FIRST_ALLOCATION);
    unsigned char * ptr2 = (unsigned char*)pws.ptr;
    for(size_t i = 0; i < FIRST_ALLOCATION; i++)
    {
        unsigned char c = i;
        BOOST_REQUIRE_EQUAL(ptr2[i], c);
    }

    //! [Allocate a larger unit of memory with the same shared memory token and fill it with data]

    unsigned char * ptr3 = (unsigned char*)create_shmanager().alloc(h, SECOND_ALLOCATION);
    create_shmanager().free(h, ptr);

    for(size_t i = 0; i < SECOND_ALLOCATION; i++)
    {
        ptr3[i] = i;
    }

    //! [Read from the larger unit of memory]
    buf->update_key(true);
    pws = buf->attach();
    buf->detach(false); // Signals that the consumer has finished reading from the previous allocation and it can now be deleted

    BOOST_REQUIRE_EQUAL(pws.size, SECOND_ALLOCATION);
    unsigned char * ptr4 = (unsigned char*)pws.ptr;
    for(size_t i = 0; i < SECOND_ALLOCATION; i++)
    {
        unsigned char c = i;
        BOOST_REQUIRE_EQUAL(ptr4[i], c);
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
	{ptr[i] = i;}
	mem.hostToDevice();

	mem.resize(0);
	BOOST_REQUIRE_EQUAL(mem.size(),0);
	BOOST_REQUIRE_EQUAL(mem.msize(),FIRST_ALLOCATION);

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

	// check that the data are retained on device

	mem.deviceToHost();
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

template<typename Mem>
void copy_device_host_test()
{
	Mem mem1;
	Mem mem2;

	mem1.allocate(300);
	mem2.allocate(500);

	if (mem1.isDeviceHostSame() == false)
	{
		BOOST_REQUIRE(mem1.getPointer() != mem2.getDevicePointer());
	}

	unsigned char * ptr1 = (unsigned char *)mem1.getPointer();
	unsigned char * ptr2 = (unsigned char *)mem2.getPointer();

	for (size_t i = 0 ; i < mem1.size() ; i++)
	{ptr1[i] = i;}

	// force to move the memory from host to device
	mem1.hostToDevice();
	mem2.copyDeviceToDevice(mem1);
	mem2.deviceToHost();

	for (size_t i = 0 ; i < mem1.size() ; i++)
	{BOOST_REQUIRE_EQUAL(ptr1[i],ptr2[i]);}
}

BOOST_AUTO_TEST_CASE( use_heap_memory )
{
	test<HeapMemory>();
#ifdef CUDA_GPU
	test<CudaMemory>();
#endif
}

BOOST_AUTO_TEST_CASE( use_shared_memory )
{
        test_shm<HeapMemory>();
}

BOOST_AUTO_TEST_CASE( shm_example )
{
        shm_usage();
}

BOOST_AUTO_TEST_CASE( use_memory )
{
	test<HeapMemory>();
#ifdef CUDA_GPU
	test<CudaMemory>();
#endif
}

BOOST_AUTO_TEST_CASE( use_bheap_memory )
{
	Btest<BMemory<HeapMemory>>();
}

BOOST_AUTO_TEST_CASE( use_bcuda_memory )
{
#ifdef CUDA_GPU
	Btest<BMemory<CudaMemory>>();
#endif
}

BOOST_AUTO_TEST_CASE( swap_heap_memory )
{
	Stest<HeapMemory>();
}

BOOST_AUTO_TEST_CASE( copy_device_host_test_use )
{
	copy_device_host_test<HeapMemory>();
#ifdef CUDA_GPU
	copy_device_host_test<CudaMemory>();
#endif
}

BOOST_AUTO_TEST_SUITE_END()


#endif /* HEAPMEMORY_UNIT_TESTS_HPP_ */
