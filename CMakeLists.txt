cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openfpm_devices LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake_modules/)

set(BOOST_INCLUDE ${Boost_INCLUDE_DIR} CACHE PATH "Include directory for BOOST")
set(HIP_ENABLE CACHE BOOL "Enable HIP compiler")

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)

find_package(Boost 1.68.0 REQUIRED COMPONENTS unit_test_framework iostreams program_options)
find_package(HIP)

if (HIP_ENABLE)
        if(NOT HIP_FOUND)
                message( FATAL_ERROR "HIP has not been found" )
        endif()
        if (HIP_ENABLE)
                set(CMAKE_CXX_EXTENSIONS OFF)
        endif()
endif()

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

