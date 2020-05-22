#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstdlib>

#include "ShmBuffer.hpp"

#define CONSEM 0   // index of semaphore for consumer
#define PROSEM 1   // index of semaphore for producer
#define KEYINIT -1 // initial value of current_key to signify no previous memory allocated
#define NEXTKEY (1^current_key)
#define PREVKEY NEXTKEY

ShmBuffer::ShmBuffer(std::string pname, int rank, bool verbose) : sems(pname, rank, verbose, false), current_key(KEYINIT), shmid(-1), verbose(verbose) // , ptr(NULL)
{
    for (int i = 0; i < NKEYS; ++i) {
        ptrs[i] = NULL;
        size[i] = -1;
    }
    // find_active();
}

ShmBuffer::~ShmBuffer()
{
    if (current_key != KEYINIT) {
        detach(true);
        detach(false);
    }
}

void ShmBuffer::find_active() // move to attach(), should always be called before it
{
    current_key = KEYINIT;
    for (int i = 0; i < NKEYS; ++i) {
        if (sems.get(i, PROSEM) > 0) { // producer using memory i

            // increment consumer semaphore
            if (sems.get(i, CONSEM) == 0) // using semaphore as mutex
                sems.incr(i, CONSEM);

            if (sems.get(i, PROSEM) == 0) {
                //The memory was already deleted before the CONSEM could be incremented
                if(verbose) std::cout << "The memory was already deleted before CONSEM could be incremented. Decrementing and looking for memory again" << std::endl;
                sems.decr(i, CONSEM);
                update_key(true);
            }
            else {
                // detach from current memory, toggle key
                // detach();
                current_key = i;
            }
            break;
        }
    }
}

ptr_with_size ShmBuffer::attach()
{
    if (ptrs[current_key] != NULL) {
        ptr_with_size ret{};
        ret.ptr = ptrs[current_key];
        ret.size = size[current_key];
        return ret;
    }

    int key = sems[current_key];

    if (verbose) std::cout << "attaching to key " << key << " with no " << current_key << std::endl; // test

    // shmget returns an identifier in shmid
    shmid = shmget(key, 0, 0);

    shmid_ds shm_id;
    shmctl(shmid, IPC_STAT, &shm_id);

    if (shmid == -1) {
        if (verbose) std::cout<< "shmget error in consumer" << std::endl;
        perror("shmget"); std::exit(1);
    }
    if (verbose) std::cout << "shmid: " << shmid << std::endl; // test

    // shmat to attach to shared memory
    ptrs[current_key] = shmat(shmid, NULL, 0);
    if (ptrs[current_key] == NULL) {
        perror("shmat"); std::exit(1);
    }

    size[current_key] = shm_id.shm_segsz;
    if (verbose) std::cout << "size of segment obtained is " << size[current_key] << std::endl; // test


    ptr_with_size ret{};
    ret.ptr = ptrs[current_key];
    ret.size = size[current_key];

    return ret;
}

void ShmBuffer::detach(bool current)
{
    int key = current ? current_key : PREVKEY;

    if (ptrs[key] == NULL)
        return;

    // detach from shared memory
    shmdt(ptrs[key]);
    ptrs[key] = NULL;

    // release semaphore, alerting producer to delete shmid
    sems.decr(key, CONSEM);
}

void ShmBuffer::update_key(bool wait) // should keep some sort of mutex for ptr, so that it is never read as null between detaching and attaching
{
    if (current_key == KEYINIT) { // called initially
        std::cout << "looking for available memory" << std::endl; // test

        do {
            find_active();
        } while (current_key == KEYINIT); // loop until an active memory segment is found

        if (verbose) std::cout << "found memory " << current_key << std::endl; // test
    } else {
        if (wait) {
            if (verbose) std::cout << "waiting for memory " << NEXTKEY << std::endl; // test
            sems.waitgeq(NEXTKEY, PROSEM, 1);
        } else {
            if (verbose) std::cout << "checking for memory " << NEXTKEY << std::endl; // test
            while (sems.get(NEXTKEY, PROSEM) == 0);
        }
        if (verbose) std::cout << "memory " << NEXTKEY << " available" << std::endl; // test

        // increment consumer semaphore
        if (sems.get(NEXTKEY, CONSEM) == 0) // using semaphore as mutex
            sems.incr(NEXTKEY, CONSEM);

        if (sems.get(NEXTKEY, PROSEM) == 0) {
            //The memory was already deleted before the CONSEM could be incremented
            if(verbose) std::cout << "The memory was already deleted before CONSEM could be incremented. Decrementing and looking for memory again" << std::endl;
            sems.decr(NEXTKEY, CONSEM);
            update_key(wait);
        }
        else {
            // detach from current memory, toggle key
            // detach();
            current_key = NEXTKEY;
        }

    }
    // assert ptr == NULL

    // attach to new key
    // attach();
}

// ignore these for now

/*

void ShmBuffer::loop()
{
	do {
		reattach(true);
		// do something with ptr
	} while (ptr != NULL); // TODO make sure to stop looping when another call to detach is made
}

void ShmBuffer::init()
{
	if (current_key == KEYINIT)
		find_active();
	out = std::async(std::launch::async, &ShmBuffer::loop, this);
}

void ShmBuffer::term()
{
	detach();
	out.wait(); // TODO use wait_for instead
}

*/