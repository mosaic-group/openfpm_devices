/*
 * ShmAllocator_manager.cpp
 *
 *  Created on: Aug 13, 2019
 *      Author: i-bird
 */

#include "ShmAllocator_manager.hpp"

ShmAllocator_manager * shm_man_sing = NULL;

ShmAllocator_manager & create_shmanager()
{
	if (shm_man_sing == NULL)
	{
		shm_man_sing = new ShmAllocator_manager();
	}

	return *shm_man_sing;
}

void destroy_shmanager()
{
	if (shm_man_sing != NULL)
	{
		delete(shm_man_sing);
	}
}
