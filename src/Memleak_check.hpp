#include "config.h"
#include <iostream>
#include <map>
#include <iomanip>

#ifndef MEMLEAK_CHECK_HPP
#define MEMLEAK_CHECK_HPP

typedef unsigned char * byte_ptr;

#ifdef SE_CLASS2

/////// POSSIBLE EVENTS /////////

#define VCLUSTER_EVENT 0x2001

#define VECTOR_EVENT 0x1102
#define VECTOR_STD_EVENT 0x1101
#define GRID_EVENT 0x1100

#define VECTOR_DIST_EVENT 0x4002
#define GRID_DIST_EVENT   0x4001

#define HEAPMEMORY_EVENT 0x0100
#define CUDAMEMORY_EVENT 0x0101

/////////////////////////////////

#include "util/se_util.hpp"
#include "ptr_info.hpp"
#include <string>

#define MEM_ERROR_OBJECT std::runtime_error("Runtime memory error")

extern long int msg_on_alloc;
extern long int msg_on_dealloc;
extern long int thr_on_alloc;
extern std::string col_stop;
extern long int new_data;
extern size_t delete_data;

extern std::map<byte_ptr,ptr_info> active_ptr;

extern long int process_v_cl;
extern long int process_to_print;

/*! \brief Check and remove the active pointer
 *
 * Check and remove the pointer from the active list
 *
 * \param pointer to check and remove
 *
 * \return true if the operation succeded, false if the pointer does not exist
 *
 */
static bool remove_ptr(const void * ptr)
{
	// Check if the pointer exist
	std::map<byte_ptr, ptr_info>::iterator it = active_ptr.find((byte_ptr)ptr);

	// if the element does not exist, print that something wrong happened and return
	if ( it == active_ptr.end() )
	{
		std::cout << "Error " << __FILE__ << ":" << __LINE__ << " pointer not found " << ptr << "\n";
		ACTION_ON_ERROR(MEM_ERROR_OBJECT);
		return false;
	}

	it->second.ref_id--;

	// erase the pointer
	if (it->second.ref_id == 0)
		active_ptr.erase((byte_ptr)ptr);

	return true;
}

#define PRJ_DEVICES 0
#define PRJ_DATA 1
#define PRJ_VCLUSTER 2
#define PRJ_IO 3
#define PRJ_PDATA 4

/*! \brief Get the color for printing unalloc pointers
 *
 * \param project_id id of the project
 * \param size size of the allocation
 * \param col color
 *
 */
inline static void get_color(size_t project_id, size_t size, std::string & col)
{
	if (size == 8)
	{
		switch (project_id)
		{
		case PRJ_DEVICES:
			col = std::string("\e[97m");
			break;
		case PRJ_DATA:
			col = std::string("\e[95m");
			break;
		case PRJ_VCLUSTER:
			col = std::string("\e[96m");
			break;
		case PRJ_IO:
			col = std::string("\e[97m");
			break;
		case PRJ_PDATA:
			col = std::string("\e[93m");
			break;
		}
	}
	else
	{
		switch (project_id)
		{
		case PRJ_DEVICES:
			col = std::string("\e[7;92m");
			break;
		case PRJ_DATA:
			col = std::string("\e[7;95m");
			break;
		case PRJ_VCLUSTER:
			col = std::string("\e[7;96m");
			break;
		case PRJ_IO:
			col = std::string("\e[7;97m");
			break;
		case PRJ_PDATA:
			col = std::string("\e[7;93m");
			break;
		}
	}
}

/*! \brief Given the structure id it convert to a human readable structure string
 *
 * \param project_id id of the project
 * \param prj string that identify the project
 *
 */
inline static void get_structure(size_t struct_id, std::string & str)
{
	switch (struct_id)
	{
		case VCLUSTER_EVENT:
			str = std::string("Vcluster");
			break;
		case VECTOR_STD_EVENT:
			str = std::string("Vector_std");
			break;
		case VECTOR_EVENT:
			str = std::string("Vector_native");
			break;
		case GRID_EVENT:
			str = std::string("Grid");
			break;
		case VECTOR_DIST_EVENT:
			str = std::string("Vector distributed");
			break;
		case GRID_DIST_EVENT:
			str = std::string("Grid distributed");
			break;
		case HEAPMEMORY_EVENT:
			str = std::string("HeapMemory");
			break;
		case CUDAMEMORY_EVENT:
			str = std::string("CudaMemory");
			break;
		default:
			str = std::to_string(struct_id);
	}
}


/*! \brief Given the project id it convert to a human readable project string
 *
 * \param project_id id of the project
 * \param prj string that identify the project
 *
 */
inline static void get_project(size_t project_id, std::string & prj)
{
	switch (project_id)
	{
	case PRJ_DEVICES:
		prj = std::string("devices");
		break;
	case PRJ_DATA:
		prj = std::string("data");
		break;
	case PRJ_VCLUSTER:
		prj = std::string("vcluster");
		break;
	case PRJ_IO:
		prj = std::string("io");
		break;
	case PRJ_PDATA:
		prj = std::string("pdata");
		break;
	}
}

/*! \brief Print all active structures
 *
 * Print all active structures
 *
 */
inline static void print_alloc()
{
	std::string col;
	std::string sid;
	std::string prj;

	for (std::map<byte_ptr,ptr_info>::iterator it = active_ptr.begin(); it != active_ptr.end(); ++it)
	{
		get_color(it->second.project_id,it->second.size,col);
		get_structure(it->second.struct_id,sid);
		get_project(it->second.project_id,prj);

		std::cout << col << "Allocated memory " << (void *)it->first << "     size=" << it->second.size << "     id=" << it->second.id << "     structure id=" << std::hex <<  sid << std::dec << "    project id=" << prj << col_stop << "\n";
	}
}

/* \brief When the allocation id==break_id is performed, print a message
 *
 * \param break_id
 *
 */
inline static void message_on_alloc(long int break_id)
{
	msg_on_alloc = break_id;
}

/* \brief When the de-allocation id==break_id is performed, print a message
 *
 * \param break_id
 *
 */
inline static void message_on_dealloc(long int break_id)
{
	msg_on_dealloc = break_id;
}

/* \brief When the allocation id==break_id is performed, throw
 *
 * \param throw_id
 *
 */
inline static void throw_on_alloc(long int throw_id)
{
	thr_on_alloc = throw_id;
}

/*! \brief Add the new allocated active pointer
 *
 * Add the new allocated active pointer
 *
 * \param new data active pointer
 * \param sz size of the new allocated memory
 *
 */
inline static bool check_new(const void * data, size_t sz, size_t struct_id, size_t project_id)
{
	// Add a new pointer
	new_data++;
	ptr_info & ptr = active_ptr[(byte_ptr)data];
	if (ptr.ref_id >= 1)
	{
		if (sz > ptr.size)	ptr.size = sz;
	}
	else
		ptr.size = sz;
	ptr.id = new_data;
	ptr.struct_id = struct_id;
	ptr.project_id = project_id;
	ptr.ref_id++;

#ifdef SE_CLASS2_VERBOSE
	if (process_to_print < 0 || process_to_print == process_v_cl)
		std::cout << "New data: " << new_data << "   " << data << "  " << ptr.size << "\n";
#endif

	if  (msg_on_alloc == new_data)
		std::cout << "Detected allocation: " << __FILE__ << ":" << __LINE__ << " id=" << msg_on_alloc << "\n";

	if (thr_on_alloc == new_data)
		throw MEM_ERROR_OBJECT;

	return true;
}

/*! \brief check and delete a pointer
 *
 * check and delete a pointer from the list of active pointers
 *
 * \param pointer data
 * \return true if the operation to delete succeed
 *
 */
inline static bool check_delete(const void * data)
{
	if (data == NULL)	return true;
	// Delete the pointer
	delete_data++;

	if (msg_on_dealloc == (long int)delete_data)
		std::cout << "Detected destruction: " << __FILE__ << ":" << __LINE__ << " id=" << msg_on_alloc << "\n";

	bool result = remove_ptr(data);

#ifdef SE_CLASS2_VERBOSE
	if (process_to_print < 0 || process_to_print == process_v_cl)
		std::cout << "Delete data: " << delete_data << "   " << data << "\n";
#endif

	return result;
}

/*! \brief check if the access is valid
 *
 * check if the access is valid
 *
 * \param ptr pointer we are going to access
 * \param size_access is the size in byte of the data we are fetching
 *
 * \return true if the pointer is valid
 *
 */
inline static bool check_valid(const void * ptr, size_t size_access)
{
	if (active_ptr.size() == 0)
	{
		std::cerr << "Error invalid pointer: " << __FILE__ << ":" << __LINE__ << "  " << ptr << "\n";
		ACTION_ON_ERROR(MEM_ERROR_OBJECT);
		return false;
	}

	// get the upper bound

	std::map<byte_ptr, ptr_info>::iterator l_b = active_ptr.upper_bound((byte_ptr)ptr);

	// if there is no memory that satisfy the request
	if (l_b == active_ptr.begin())
	{
		if (process_to_print < 0 || process_to_print == process_v_cl)
		{
			std::cerr << "Error invalid pointer: " << __FILE__ << ":" << __LINE__ << "  " << ptr << "   base allocation id=" << l_b->second.id << "\n";
			ACTION_ON_ERROR(MEM_ERROR_OBJECT);
		}
		return false;
	}

	//! subtract one
	l_b--;

	// if there is no memory that satisfy the request
	if (l_b == active_ptr.end())
	{
		if (process_to_print < 0 || process_to_print == process_v_cl)
		{
			std::cerr << "Error invalid pointer: " << __FILE__ << ":" << __LINE__ << "  " << ptr << "   base allocation id=" << l_b->second.id  << "\n";
			ACTION_ON_ERROR(MEM_ERROR_OBJECT);
		}
		return false;
	}

	// check if ptr is in the range

	size_t sz = l_b->second.size;

	if (((unsigned char *)l_b->first) + sz < ((unsigned char *)ptr) + size_access)
	{
		bool found = false;

		// Here we do a full search across all the registered pointers

		std::map<byte_ptr, ptr_info>::iterator fit = active_ptr.begin();
		for(; fit != active_ptr.end(); fit++)
		{
			if (ptr >= fit->first  && (((unsigned char *)ptr) + size_access) <= (((unsigned char *)fit->first) + fit->second.size) )
			{
				found = true;
				break;
			}
		}

		if (found == false)
		{
			if (process_to_print < 0 || process_to_print == process_v_cl)
			{
				std::cerr << "Error invalid pointer: " << __FILE__ << ":" << __LINE__ << "  "  << ptr << " base  allocation id=" << l_b->second.id << "\n";
				ACTION_ON_ERROR(MEM_ERROR_OBJECT);
			}
		}
	}

	return true;
}


/*! \brief check if the access is valid
 *
 * check if the access is valid
 *
 * \param ptr pointer we are going to access
 * \param size_access is the size in byte of the data we are fetching
 *
 * \return true if the pointer is valid
 *
 */
inline static long int check_whoami(const void * ptr, size_t size_access)
{
	if (active_ptr.size() == 0)
		return -1;

	// get the upper bound

	std::map<byte_ptr, ptr_info>::iterator l_b = active_ptr.upper_bound((byte_ptr)ptr);

	// if there is no memory that satisfy the request
	if (l_b == active_ptr.begin())
		return -1;

	//! subtract one
	l_b--;

	// if there is no memory that satisfy the request
	if (l_b == active_ptr.end())
		return -1;

	// check if ptr is in the range

	size_t sz = l_b->second.size;

	if (((unsigned char *)l_b->first) + sz < ((unsigned char *)ptr) + size_access)
		return -1;

	return l_b->second.id;
}

/*! \brief In case of Parallel application it set the process that print the
 *
 * \param p_to_print is < 0 (Mean all)
 *
 */
inline static void set_process_to_print(long int p_to_print)
{
	process_to_print = p_to_print;
}

#else



#endif
#endif
