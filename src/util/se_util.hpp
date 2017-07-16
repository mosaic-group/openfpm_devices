/*
 * se_util.hpp
 *
 *  Created on: Oct 22, 2015
 *      Author: i-bird
 */

#ifndef OPENFPM_DATA_SRC_UTIL_SE_UTIL_HPP_
#define OPENFPM_DATA_SRC_UTIL_SE_UTIL_HPP_

#include "print_stack.hpp"

// Macro that decide what to do in case of error
#ifdef STOP_ON_ERROR
#define ACTION_ON_ERROR(error) print_stack();abort();
#define THROW noexcept(true)
#elif defined(THROW_ON_ERROR)
#define ACTION_ON_ERROR(error) if (!std::uncaught_exception()) {print_stack();throw error;}
#define THROW noexcept(false)
#else
#define ACTION_ON_ERROR(error) print_stack();
#define THROW noexcept(true)
#endif


#endif /* OPENFPM_DATA_SRC_UTIL_SE_UTIL_HPP_ */
