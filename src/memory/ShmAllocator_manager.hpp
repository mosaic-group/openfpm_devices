/*
 * ShmAllocator_manager.hpp
 *
 *  Created on: Aug 13, 2019
 *      Author: i-bird
 */

#ifndef SHMALLOCATOR_MANAGER_HPP_
#define SHMALLOCATOR_MANAGER_HPP_

#include <vector>
#include "ShmAllocator.hpp"

struct handle_shmem
{
	int id;
};

class ShmAllocator_manager
{

	std::vector<ShmAllocator *> mems;

public:

	~ShmAllocator_manager()
	{
		for (int i = 0 ; i < mems.size() ; i++)
		{
			handle_shmem hs;
			hs.id = i;
			destroy(hs);
		}
	}

	/*! \brief Allocate new shared memory
	 *
	 * \param name shared memory name
	 * \param rank processor rank
	 *
	 */
	handle_shmem create(const std::string & name, int rank)
	{
		mems.push_back(NULL);
		mems[mems.size()-1] = new ShmAllocator(name,rank, true);

		handle_shmem hs;
		hs.id = mems.size()-1;

		return hs;
	}

	/*! \brief Allocate shared memory
	 *
	 * \param Allocator handle
	 * \param size of the memory to allocate
	 *
	 */
	void * alloc(handle_shmem handle, size_t size)
	{
		return mems[handle.id]->shm_alloc(size);
	}

	/*! \brief Free shared memory
	 *
	 * \param handle
	 * \param ptr pointer to the shared memory
	 *
	 */
	void free(handle_shmem handle, void * ptr)
	{
		mems[handle.id]->shm_free(ptr);
	}

	/*! \brief Destroy shared allocator
	 *
	 * \param handle
	 *
	 */
	void destroy(handle_shmem handle)
	{
	    if(handle.id != -1)
        {
            delete(mems[handle.id]);
        }
	}
};

ShmAllocator_manager & create_shmanager();
void destroy_shmanager();

#endif /* SHMALLOCATOR_MANAGER_HPP_ */
