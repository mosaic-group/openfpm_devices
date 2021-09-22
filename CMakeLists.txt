cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openfpm_devices LANGUAGES C CXX)

set(BOOST_INCLUDE ${Boost_INCLUDE_DIR} CACHE PATH "Include directory for BOOST")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake_modules/)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CUDA_STANDARD 14)

find_package(Boost 1.72.0 REQUIRED COMPONENTS unit_test_framework iostreams program_options OPTIONAL_COMPONENTS fiber context)
find_package(OpenMP)

set(CUDA_ON_CPU CACHE BOOL "Make Cuda work on heap")

# Save Boost_LIBRARIES before alpaka fuck-up Boost
set(Boost_LIBRARIES_BCK ${Boost_LIBRARIES})

if(ENABLE_GPU)
	enable_language(CUDA)
	find_package(CUDA)
	if (CUDA_VERSION_MAJOR EQUAL 9 AND CUDA_VERSION_MINOR EQUAL 2)
		message("CUDA is compatible 9.2")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=611 --diag_suppress=2885 --diag_suppress=2886  --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111" --expt-extended-lambda PARENT_SCOPE)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=611 --diag_suppress=2885 --diag_suppress=2886  --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111\" --expt-extended-lambda " PARENT_SCOPE)
	elseif ( CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR EQUAL 1 )
        message("CUDA is compatible 10.1")
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=2931 --diag_suppress=2930 --diag_suppress=2929 --diag_suppress=2928  --diag_suppress=2915 --diag_suppress=2912 --diag_suppress=2913 --diag_suppress=111 --diag_suppress=186 --diag_suppress=611 --diag_suppress=128" --expt-extended-lambda PARENT_SCOPE)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT "-Xcudafe \"--display_error_number --display_error_number --diag_suppress=2931 --diag_suppress=2930 --diag_suppress=2929 --diag_suppress=2928  --diag_suppress=2915 --diag_suppress=2912 --diag_suppress=2913 --diag_suppress=111 --diag_suppress=186 --diag_suppress=611 --diag_suppress=128 \" --expt-extended-lambda" PARENT_SCOPE)
    elseif ( CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR EQUAL 2 )
        message("CUDA is compatible 10.2")
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=2976 --diag_suppress=2977 --diag_suppress=2978 --diag_suppress=2979 --diag_suppress=1835 --diag_suppress=611 --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda PARENT_SCOPE)
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=2976 --diag_suppress=2977 --diag_suppress=2978 --diag_suppress=2979 --diag_suppress=1835 --diag_suppress=611 --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda" PARENT_SCOPE)
    elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 0 )
        message("CUDA is compatible 11.0")
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=3056 --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda PARENT_SCOPE)
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=3056 --diag_suppress=3057  --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda" PARENT_SCOPE)
	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 1 )
        message("CUDA is compatible 11.1")
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=3124 --diag_suppress=3126 --diag_suppress=3125 --diag_suppress=3123 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda PARENT_SCOPE)
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=3124 --diag_suppress=3126  --diag_suppress=3125 --diag_suppress=3123 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda" PARENT_SCOPE)
        elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 2 )
        message("CUDA is compatible 11.2")
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=20014 --diag_suppress=20013 --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=611 --diag_suppress=550  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda PARENT_SCOPE)
        set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=20014 --diag_suppress=20013 --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=611 --diag_suppress=550  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda" PARENT_SCOPE)
	else()
		message(FATAL_ERROR "CUDA is incompatible, version 9.2 10.1 10.2 11.1 and 11.2 is only supported")
	endif()
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

