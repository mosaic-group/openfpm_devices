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
#ifndef __CYGWIN__ 
#include <execinfo.h>
#endif

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
#if defined(PRINT_STACKTRACE) && !defined(__CYGWIN__)

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

	std::string translators[]={"eu-addr2line","addr2line"};
	std::string translator;

	for (size_t i = 0 ; i < sizeof(translators)/sizeof(std::string) ; i++)
	{
		// for for the best address to source code translator
		char syscom[256];
		sprintf(syscom,"%s --version",translators[i].c_str());

		std::string ss = exec(syscom);
		size_t found = ss.find("command not found");
		if (found == std::string::npos)
		{
			// command exist
			translator = translators[i];
			str << "Using translator: " << translator << std::endl;
			break;
		}
	}

	if (translator.size() == 0)
	{
		str << "In order to have a more detailed stack-trace with function name and precise location in the source code" <<
			"Please install one of the following utils: ";

		str << translators[0];

		for (size_t i = 1 ; i < sizeof(translators)/sizeof(std::string) ; i++)
			str << "," << translators[i];
	}

	for (int i = 0 ; i < ncall ; i++)
	{
		str << "\033[1m" << "CALL(" << i << ")" << "\033[0m" <<  " Address: " << trace[i] << "   " << messages[i] << "   ";

		if (translator.size() != 0)
		{
			char syscom[256];
			sprintf(syscom,"%s %p -f --demangle -e %s",translator.c_str(), trace[i],program_name.c_str()); //last parameter is the name of this app

			std::string ss = exec(syscom);
			std::stringstream sss(ss);
			std::string sfunc;
			std::string sloc;
			std::getline(sss,sfunc,'\n');
			std::getline(sss,sloc,'\n');
			str << std::endl;
			str << "\033[35m" << "Function:" << std::endl << sfunc << "\033[0m" << std::endl;
			str << "\033[1;31m" << "Location:" << std::endl << sloc << "\033[0m" << std::endl;
		}
		else
		{
			str << std::endl;
		}
	}

	std::cerr << str.str() << std::endl;

#else

	std::cerr << "Stack trace deactivated, use #define PRINT_STACKTRACE to activate" << std::endl;

#endif
}



#endif /* OPENFPM_DATA_SRC_UTIL_PRINT_STACK_HPP_ */
