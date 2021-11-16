cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openfpm_devices LANGUAGES C CXX)

set(BOOST_INCLUDE ${Boost_INCLUDE_DIR} CACHE PATH "Include directory for BOOST")
set(AMD_ARCH_COMPILE "gfx900" CACHE STRING "AMD gpu architecture used to compile kernels")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake_modules/ /opt/rocm/hip/cmake)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CUDA_STANDARD 14)

set (ENV{BOOST_ROOT} ${BOOST_ROOT})
set (Boost_NO_BOOST_CMAKE OFF)
find_package(Boost 1.72.0 REQUIRED COMPONENTS unit_test_framework iostreams program_options OPTIONAL_COMPONENTS fiber context)
find_package(OpenMP)

set(CUDA_ON_BACKEND CACHE STRING "Activate several backend for CUDA")

# Save Boost_LIBRARIES before alpaka fuck-up Boost
set(Boost_LIBRARIES_BCK ${Boost_LIBRARIES})

if (CUDA_ON_BACKEND STREQUAL "HIP" AND NOT HIP_FOUND)
	find_package(HIP)
        if(NOT HIP_FOUND)
                message( FATAL_ERROR "HIP has not been found" )
	else()
		message("HIP has been found")
        endif()
        if (CUDA_ON_BACKEND="HIP")
                set(CMAKE_CXX_EXTENSIONS OFF)
        endif()
        set(DEFINE_CUDA_GPU "#define CUDA_GPU")
endif()

if (CUDA_ON_BACKEND STREQUAL "CUDA")
	enable_language(CUDA)
	include(CheckLanguage)
	find_package(CUDA)
	if (CUDA_VERSION_MAJOR EQUAL 9 AND CUDA_VERSION_MINOR EQUAL 2)
		message("CUDA is compatible 9.2")
	elseif ( CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR EQUAL 1 )
        	message("CUDA is compatible 10.1")
	elseif ( CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR EQUAL 2 )
		message("CUDA is compatible 10.2")
    	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 0 )
        	message("CUDA is compatible 11.0")
	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 1 )
        	message("CUDA is compatible 11.1")
	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND (CUDA_VERSION_MINOR EQUAL 2 OR CUDA_VERSION_MINOR EQUAL 3 OR CUDA_VERSION_MINOR EQUAL 4 ))
        	message("CUDA is compatible 11.2/3/4")
	else()
		message(FATAL_ERROR "CUDA is incompatible, version 9.2 10.1 10.2 11.1 11.2 11.3 11.4 is only supported")
	endif()
	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_CUDA")
        set(DEFINE_CUDA_GPU "#define CUDA_GPU")
elseif (CUDA_ON_BACKEND STREQUAL "OpenMP")
	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_OPENMP")
        set(DEFINE_CUDA_GPU "#define CUDA_GPU")
elseif (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL")
	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_SEQUENTIAL")
        set(DEFINE_CUDA_GPU "#define CUDA_GPU")
else()
	find_package(alpaka)
endif()

###### CONFIG.h FILE ######

if(HIP_FOUND)
        set(DEFINE_HIP_GPU "#define HIP_GPU")
        set(DEFINE_CUDIFY_USE_HIP "#define CUDIFY_USE_HIP")
endif()


if (Boost_FOUND)
        set(DEFINE_HAVE_BOOST "#define HAVE_BOOST")
        set(DEFINE_HAVE_BOOST_IOSTREAMS "#define HAVE_BOOST_IOSTREAMS")
        set(DEFINE_HAVE_BOOST_PROGRAM_OPTIONS "#define HAVE_BOOST_PROGRAM_OPTIONS")
        set(DEFINE_HAVE_BOOST_UNIT_TEST_FRAMEWORK "#define HAVE_BOOST_UNIT_TEST_FRAMEWORK")

	if (Boost_CONTEXT_FOUND)
		set(DEFINE_HAVE_BOOST_CONTEXT "#define HAVE_BOOST_CONTEXT")
		set(OPTIONAL_BOOST_LIBS "-lboost_context")
	endif()
	if (Boost_FIBER_FOUND)
		set(DEFINE_HAVE_BOOST_FIBER "#define HAVE_BOOST_FIBER")
		string(CONCAT OPTIONAL_BOOST_LIBS ${OPTIONAL_BOOST_LIBS}  " -lboost_fiber")
	endif()
endif()

if (OPENMP_FOUND)
	set(DEFINE_HAVE_OPENMP "#define HAVE_OPENMP")
endif()

if (ALPAKA_ROOT)
	set(DEFINE_HAVE_ALPAKA "#define HAVE_ALPAKA")
endif()

######## COMPONENT CHECK ########, some component require dependencies, HERE we check that what you want can be satisfied

if (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL")
	## We need BOOST_CONTEXT
	if (NOT Boost_CONTEXT_FOUND)
		message(FATAL_ERROR "Error to run CUDA on CPU (SEQUENTIAL backend) we need Boost::context installed, it seems is not available in your boost installation")
	endif()
endif()

if (CUDA_ON_BACKEND STREQUAL "OpenMP")
	## We need OpenMP
	if (NOT OPENMP_FOUND OR NOT Boost_CONTEXT_FOUND)
		message(FATAL_ERROR "Error to run CUDA on CPU (OpenMP backend) we need Boost::context and OpenMPinstalled, it seems is not available in your system")
        endif()
endif()

if (CUDA_ON_BACKEND STREQUAL "HIP")
	## We need HIP
	if (NOT HIP_FOUND)
		message(FATAL_ERROR "Error to run CUDA on HIP (HIP backend) RocM/HIP, it seems is not available in your system")
        endif()
endif()

if (CUDA_ON_BACKEND STREQUAL "CUDA")
	## We need CUDA
	if (NOT CUDA_FOUND)
		message(FATAL_ERROR "Error to run on CUDA you need a CUDA compatible compiler, it seems is not available in your system")
        endif()
endif()


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/config/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config.h)

add_subdirectory (src)

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
	set(DEFINE_CUDA_GPU ${DEFINE_CUDA_GPU} PARENT_SCOPE)
	set(WARNING_SUPPRESSION_AND_OPTION_NVCC ${WARNING_SUPPRESSION_AND_OPTION_NVCC}  PARENT_SCOPE)
	set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT ${WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT} PARENT_SCOPE)
        set(DEFINE_HIP_GPU ${DEFINE_HIP_GPU} PARENT_SCOPE)
        set(DEFINE_CUDIFY_USE_HIP ${DEFINE_CUDIFY_USE_HIP} PARENT_SCOPE)
        set(DEFINE_HAVE_BOOST ${DEFINE_HAVE_BOOST} PARENT_SCOPE)
        set(DEFINE_HAVE_BOOST_IOSTREAMS ${DEFINE_HAVE_BOOST_IOSTREAMS} PARENT_SCOPE)
        set(DEFINE_HAVE_BOOST_PROGRAM_OPTIONS ${DEFINE_HAVE_BOOST_PROGRAM_OPTIONS} PARENT_SCOPE)
        set(DEFINE_HAVE_BOOST_UNIT_TEST_FRAMEWORK ${DEFINE_HAVE_BOOST_UNIT_TEST_FRAMEWORK} PARENT_SCOPE)
	set(DEFINE_HAVE_BOOST_CONTEXT ${DEFINE_HAVE_BOOST_CONTEXT} PARENT_SCOPE)
	set(DEFINE_HAVE_BOOST_FIBER ${DEFINE_HAVE_BOOST_FIBER} PARENT_SCOPE)
        set(DEFINE_CUDIFY_BACKEND ${DEFINE_CUDIFY_BACKEND} PARENT_SCOPE)
        set(DEFINE_CUDA_GPU ${DEFINE_CUDA_GPU}  PARENT_SCOPE)
	set(OPTIONAL_BOOST_LIBS ${OPTIONAL_BOOST_LIBS} PARENT_SCOPE)
	set(DEFINE_HAVE_OPENMP ${DEFINE_HAVE_OPENMP} PARENT_SCOPE)
	set(DEFINE_HAVE_ALPAKA ${DEFINE_HAVE_ALPAKA} PARENT_SCOPE)
endif()

