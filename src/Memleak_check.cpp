#include "config.h"
#include "Memleak_check.hpp"
#include "ptr_info.hpp"

// counter for allocation of new memory
size_t new_data;

// counter to delete memory
size_t delete_data;

// structure that store all the active pointer
std::map<byte_ptr, ptr_info> active_ptr;

// Running process id
long int process_v_cl;

// Process to print
long int process_to_print = 0;

// A way to stop the color
std::string col_stop("\e[0m");

// Print a message when allocation with id==msg_on_alloc is performed
long int msg_on_alloc = -1;

// throw when allocation with id==throw_on_alloc is performed
long int thr_on_alloc = -1;
