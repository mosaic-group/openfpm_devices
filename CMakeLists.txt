cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openfpm_devices LANGUAGES CXX)


set(BOOST_INCLUDE ${Boost_INCLUDE_DIR} CACHE PATH "Include directory for BOOST")

set (CMAKE_CXX_STANDARD 11)

find_package(Boost 1.52.0 REQUIRED unit_test_framework iostreams program_options)


if(ENABLE_GPU)
	enable_language(CUDA)
	find_package(CUDA)
endif()

###### CONFIG.h FILE ######

if(CUDA_FOUND)
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
	set(CUDA_SOURCES memory/CudaMemory.cu)
endif()

if (BOOST_FOUND)
	set(DEFINE_HAVE_BOOST "#define HAVE_BOOST")
	set(DEFINE_HAVE_BOOST_IOSTREAMS "#define HAVE_BOOST_IOSTREAMS")
	set(DEFINE_HAVE_BOOST_PROGRAM_OPTIONS "#define HAVE_BOOST_PROGRAM_OPTIONS")
	set(DEFINE_HAVE_BOOST_UNIT_TEST_FRAMEWORK "#define HAVE_BOOST_UNIT_TEST_FRAMEWORK")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/config/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config.h)

add_subdirectory (src)

