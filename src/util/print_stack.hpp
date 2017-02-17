/*
 * print_stack.hpp
 *
 *  Created on: Feb 12, 2017
 *      Author: i-bird
 */

#ifndef OPENFPM_DATA_SRC_UTIL_PRINT_STACK_HPP_
#define OPENFPM_DATA_SRC_UTIL_PRINT_STACK_HPP_

#include <sstream>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <execinfo.h>

extern std::string program_name;

/*! \brief Execute a command getting the std::cout
 *
 *
 */
static inline std::string exec(const char* cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get()))
    {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

/*! \brief Print the stack trace
 *
 *
 */
static inline void print_stack()
{
#ifdef PRINT_STACKTRACE

	void *trace[256];

	int ncall = backtrace(trace,256);
	char ** messages = backtrace_symbols(trace, ncall);

	std::stringstream str;

	str << "\033[1;31m*******************************************\033[0m" << std::endl;
	str << "\033[1;31m*********** STACK TRACE *******************\033[0m" << std::endl;
	str << "\033[1;31m*******************************************\033[0m" << std::endl;

	str << "\033[1mThe stack trace indicate where in the code happened the error the error " <<
			     "is in general detected inside the library. In order to find the position " <<
			     "in your code, follow from top to bottom the list of calls until you find " <<
				 "a source code familiar to you.\033[0m" << std::endl;

	for (int i = 0 ; i < ncall ; i++)
	{
		str << "STACK TRACE Address: " << trace[i] << "   " << messages[i] << "   source:";

		char syscom[256];
		sprintf(syscom,"addr2line %p -e %s", trace[i],program_name.c_str()); //last parameter is the name of this app

		std::string ss = exec(syscom);
		str << std::endl << "\033[1;31m" << ss << "\033[0m";
	}

	std::cerr << str.str() << std::endl;

#endif
}



#endif /* OPENFPM_DATA_SRC_UTIL_PRINT_STACK_HPP_ */
