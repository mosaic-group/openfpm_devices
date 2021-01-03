cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openfpm_devices LANGUAGES C CXX)

set(BOOST_INCLUDE ${Boost_INCLUDE_DIR} CACHE PATH "Include directory for BOOST")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake_modules/)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CUDA_STANDARD 14)

find_package(Boost 1.72.0 REQUIRED COMPONENTS unit_test_framework iostreams program_options OPTIONAL_COMPONENTS fiber context)

set(CUDA_ON_CPU CACHE BOOL "Make Cuda work on heap")

# Save Boost_LIBRARIES before alpaka fuck-up Boost
set(Boost_LIBRARIES_BCK ${Boost_LIBRARIES})

if(ENABLE_GPU)
	enable_language(CUDA)
	find_package(CUDA)
else()
	find_package(alpaka)
endif()

###### CONFIG.h FILE ######

if(CUDA_FOUND)
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
	set(CUDA_SOURCES memory/CudaMemory.cu)
endif()

if(CUDA_ON_CPU)
        set(DEFINE_CUDA_GPU "#define CUDA_GPU")
endif()

if (BOOST_FOUND)
	set(DEFINE_HAVE_BOOST "#define HAVE_BOOST")
	set(DEFINE_HAVE_BOOST_IOSTREAMS "#define HAVE_BOOST_IOSTREAMS")
	set(DEFINE_HAVE_BOOST_PROGRAM_OPTIONS "#define HAVE_BOOST_PROGRAM_OPTIONS")
	set(DEFINE_HAVE_BOOST_UNIT_TEST_FRAMEWORK "#define HAVE_BOOST_UNIT_TEST_FRAMEWORK")
	if (Boost_CONTEXT_FOUND)
		set(DEFINE_HAVE_BOOST_CONTEXT "#define HAVE_BOOST_CONTEXT")
	endif()
	if (Boost_FIBER_FOUND)
		set(DEFINE_HAVE_BOOST_FIBER "#define HAVE_BOOST_FIBER")
	endif()
endif()

if (ALPAKA_ROOT)
	set(DEFINE_HAVE_ALPAKA "#define HAVE_ALPAKA")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/config/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config.h)

add_subdirectory (src)

