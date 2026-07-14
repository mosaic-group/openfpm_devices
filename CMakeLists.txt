cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openfpm_devices LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
	enable_testing()
	if (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL" OR CUDA_ON_BACKEND STREQUAL "OpenMP")
		find_package(Boost 1.72 REQUIRED COMPONENTS unit_test_framework context)
	else()
		find_package(Boost 1.72 REQUIRED COMPONENTS unit_test_framework)
	endif()
endif()

if (ENABLE_GARBAGE_INJECTOR)
	set(DEFINE_GARBAGE_INJECTOR "#define GARBAGE_INJECTOR")
endif()

if (OPENMP_FOUND)
	set(DEFINE_HAVE_OPENMP "#define HAVE_OPENMP")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(CUDA_ON_CPU)
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
endif()

if (CUDA_ON_BACKEND STREQUAL "HIP")
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

if(HIP_FOUND)
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
	set(DEFINE_CUDIFY_USE_HIP "#define CUDIFY_USE_HIP")
endif()

if (CUDA_ON_BACKEND STREQUAL "CUDA")
	include(CheckLanguage)
	if (CMAKE_CUDA_COMPILER)
		set(CUDA_FOUND ON)
	endif()
	string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
	list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
	list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
	list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)
	if (CUDA_VERSION_MAJOR EQUAL 9 AND CUDA_VERSION_MINOR EQUAL 2)
		message("CUDA is compatible 9.2")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=611 --diag_suppress=2885 --diag_suppress=2886  --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=611 --diag_suppress=2885 --diag_suppress=2886  --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111\" --expt-extended-lambda ")
	elseif ( CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR EQUAL 1 )
		message("CUDA is compatible 10.1")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=2931 --diag_suppress=2930 --diag_suppress=2929 --diag_suppress=2928  --diag_suppress=2915 --diag_suppress=2912 --diag_suppress=2913 --diag_suppress=111 --diag_suppress=186 --diag_suppress=611 --diag_suppress=128" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT "-Xcudafe \"--display_error_number --display_error_number --diag_suppress=2931 --diag_suppress=2930 --diag_suppress=2929 --diag_suppress=2928  --diag_suppress=2915 --diag_suppress=2912 --diag_suppress=2913 --diag_suppress=111 --diag_suppress=186 --diag_suppress=611 --diag_suppress=128 \" --expt-extended-lambda")
	elseif ( CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR EQUAL 2 )
		message("CUDA is compatible 10.2")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=2976 --diag_suppress=2977 --diag_suppress=2978 --diag_suppress=2979 --diag_suppress=1835 --diag_suppress=611 --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=2976 --diag_suppress=2977 --diag_suppress=2978 --diag_suppress=2979 --diag_suppress=1835 --diag_suppress=611 --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda")
	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 0 )
		message("CUDA is compatible 11.0")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=3056 --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=3056 --diag_suppress=3057  --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda")
	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR EQUAL 1 )
		message("CUDA is compatible 11.1")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=3124 --diag_suppress=3126 --diag_suppress=3125 --diag_suppress=3123 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=3124 --diag_suppress=3126  --diag_suppress=3125 --diag_suppress=3123 --diag_suppress=611  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda")
	elseif ( CUDA_VERSION_MAJOR EQUAL 11 AND (CUDA_VERSION_MINOR EQUAL 2 OR CUDA_VERSION_MINOR EQUAL 3 OR CUDA_VERSION_MINOR EQUAL 4 OR CUDA_VERSION_MINOR EQUAL 5 OR CUDA_VERSION_MINOR EQUAL 6 OR CUDA_VERSION_MINOR EQUAL 7 OR CUDA_VERSION_MINOR EQUAL 8))
		message("CUDA is compatible 11.2/3/4/5")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=20014 --diag_suppress=20013 --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=611 --diag_suppress=550  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=20014 --diag_suppress=20013 --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=611 --diag_suppress=550  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda")
		elseif ( CUDA_VERSION_MAJOR EQUAL 12 AND (CUDA_VERSION_MINOR EQUAL 0 OR CUDA_VERSION_MINOR EQUAL 1 OR CUDA_VERSION_MINOR EQUAL 2 OR CUDA_VERSION_MINOR EQUAL 3 OR CUDA_VERSION_MINOR EQUAL 4 OR CUDA_VERSION_MINOR EQUAL 5 OR CUDA_VERSION_MINOR EQUAL 6))
		message("CUDA is compatible 12")
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC  -Xcudafe "--display_error_number --diag_suppress=20014 --diag_suppress=20013 --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=611 --diag_suppress=550  --diag_suppress=186 --diag_suppress=128" --expt-extended-lambda)
		set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT  "-Xcudafe \"--display_error_number --diag_suppress=20014 --diag_suppress=20013 --diag_suppress=20012 --diag_suppress=20011 --diag_suppress=611 --diag_suppress=550  --diag_suppress=186 --diag_suppress=128\" --expt-extended-lambda")
	else()
		message(FATAL_ERROR "CUDA is incompatible, version 9.2 10.1 10.2 11.1 11.2 11.3 11.4 11.5 11.6 11.7 11.8 12.0 12.1 12.2 12.3 12.4 12.5 is only supported")
	endif()
	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_CUDA")
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
elseif (CUDA_ON_BACKEND STREQUAL "OpenMP")
	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_OPENMP")
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
elseif (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL")
	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_SEQUENTIAL")
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")
elseif (CUDA_ON_BACKEND STREQUAL "METAL")
	if (NOT APPLE)
		message(FATAL_ERROR "METAL backend requires macOS/Apple platform")
	endif()

	set(DEFINE_CUDIFY_BACKEND "#define CUDIFY_USE_METAL")
	set(DEFINE_CUDA_GPU "#define CUDA_GPU")

	set(MOLTENVK_ROOT "" CACHE PATH "MoltenVK installation prefix")
	set(VULKAN_HEADERS_ROOT "" CACHE PATH "Vulkan-Headers installation prefix")
	# A prior non-Metal configure may have exported these as empty INTERNAL
	# cache entries below. Clear only unusable results so find_* searches again
	# when an existing build directory is switched to the Metal backend.
	if (NOT VULKAN_INCLUDE_DIR)
		unset(VULKAN_INCLUDE_DIR CACHE)
	endif()
	if (NOT MOLTENVK_LIBRARY)
		unset(MOLTENVK_LIBRARY CACHE)
	endif()
	find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h
		HINTS "${VULKAN_HEADERS_ROOT}/include" "${MOLTENVK_ROOT}/include"
			/opt/homebrew/opt/vulkan-headers/include /usr/local/include)
	find_library(MOLTENVK_LIBRARY MoltenVK
		HINTS "${MOLTENVK_ROOT}/lib" /opt/homebrew/opt/molten-vk/lib /usr/local/lib)
	if (NOT VULKAN_INCLUDE_DIR OR NOT MOLTENVK_LIBRARY)
		message(FATAL_ERROR "METAL backend requires Vulkan-Headers and MoltenVK")
	endif()

	set(OPENFPM_MOLTENVK_HIP_CLANG "" CACHE FILEPATH
		"HIP-capable clang built with SPIR-V support")
	set(OPENFPM_MOLTENVK_LLVM_OPT "" CACHE FILEPATH
		"llvm-opt matching the Metal HIP compiler")
	set(OPENFPM_MOLTENVK_LLVM_LINK "" CACHE FILEPATH
		"llvm-link matching the Metal HIP compiler")
	set(OPENFPM_MOLTENVK_LLVM_CONFIG "" CACHE FILEPATH
		"llvm-config matching the Metal HIP compiler")
	set(OPENFPM_MOLTENVK_CLSPV "" CACHE FILEPATH "clspv executable")
	set(OPENFPM_MOLTENVK_ABI_TOOL "" CACHE FILEPATH
		"OpenFPM packed Metal kernel ABI normalizer")
	set(OPENFPM_CHIPSTAR_ROOT "" CACHE PATH "chipStar source or install prefix")
	set(OPENFPM_CHIPSTAR_GENERATED_INCLUDE "" CACHE PATH
		"Optional chipStar generated include directory")
	if (NOT OPENFPM_MOLTENVK_CLSPV)
		find_program(_openfpm_clspv NAMES clspv
			HINTS "$ENV{CLSPV_ROOT}" PATH_SUFFIXES bin build/bin)
		if (_openfpm_clspv)
			set(OPENFPM_MOLTENVK_CLSPV "${_openfpm_clspv}"
				CACHE FILEPATH "" FORCE)
		endif()
	endif()
	if (OPENFPM_MOLTENVK_CLSPV)
		get_filename_component(_openfpm_clspv_bin
			"${OPENFPM_MOLTENVK_CLSPV}" DIRECTORY)
		get_filename_component(_openfpm_clspv_build
			"${_openfpm_clspv_bin}" DIRECTORY)
	endif()
	if (NOT OPENFPM_MOLTENVK_HIP_CLANG)
		find_program(_openfpm_spirv_clang NAMES clang++
			HINTS "$ENV{OPENFPM_MOLTENVK_LLVM_ROOT}"
				"${_openfpm_clspv_build}/third_party/llvm"
			PATH_SUFFIXES bin NO_DEFAULT_PATH)
		if (_openfpm_spirv_clang)
			set(OPENFPM_MOLTENVK_HIP_CLANG "${_openfpm_spirv_clang}"
				CACHE FILEPATH "" FORCE)
		endif()
	endif()
	if (OPENFPM_MOLTENVK_LLVM_OPT)
		get_filename_component(_openfpm_llvm_bin
			"${OPENFPM_MOLTENVK_LLVM_OPT}" DIRECTORY)
	elseif (OPENFPM_MOLTENVK_HIP_CLANG)
		get_filename_component(_openfpm_llvm_bin
			"${OPENFPM_MOLTENVK_HIP_CLANG}" DIRECTORY)
		if (EXISTS "${_openfpm_llvm_bin}/opt")
			set(OPENFPM_MOLTENVK_LLVM_OPT "${_openfpm_llvm_bin}/opt"
				CACHE FILEPATH "" FORCE)
		endif()
	endif()
	if (_openfpm_llvm_bin)
		if (NOT OPENFPM_MOLTENVK_LLVM_LINK AND
			EXISTS "${_openfpm_llvm_bin}/llvm-link")
			set(OPENFPM_MOLTENVK_LLVM_LINK
				"${_openfpm_llvm_bin}/llvm-link" CACHE FILEPATH "" FORCE)
		endif()
		if (NOT OPENFPM_MOLTENVK_LLVM_CONFIG AND
			EXISTS "${_openfpm_llvm_bin}/llvm-config")
			set(OPENFPM_MOLTENVK_LLVM_CONFIG
				"${_openfpm_llvm_bin}/llvm-config" CACHE FILEPATH "" FORCE)
		endif()
	endif()
	if (NOT OPENFPM_CHIPSTAR_ROOT AND DEFINED ENV{CHIPSTAR_ROOT} AND
		EXISTS "$ENV{CHIPSTAR_ROOT}/HIP/include")
		set(OPENFPM_CHIPSTAR_ROOT "$ENV{CHIPSTAR_ROOT}"
			CACHE PATH "" FORCE)
	endif()
	foreach(tool OPENFPM_MOLTENVK_HIP_CLANG OPENFPM_MOLTENVK_LLVM_OPT
			OPENFPM_MOLTENVK_LLVM_LINK OPENFPM_MOLTENVK_LLVM_CONFIG
			OPENFPM_MOLTENVK_CLSPV OPENFPM_CHIPSTAR_ROOT)
		if (NOT ${tool})
			message(FATAL_ERROR "${tool} is required for the METAL kernel translation layer")
		endif()
	endforeach()

	if (NOT OPENFPM_MOLTENVK_ABI_TOOL OR
		OPENFPM_MOLTENVK_ABI_TOOL STREQUAL
			"${CMAKE_BINARY_DIR}/openfpm-tools/openfpm-moltenvk-kernel-abi")
		execute_process(COMMAND "${OPENFPM_MOLTENVK_LLVM_CONFIG}" --cxxflags
			OUTPUT_VARIABLE _openfpm_llvm_cxxflags
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE _openfpm_llvm_cxxflags_result
			ERROR_VARIABLE _openfpm_llvm_cxxflags_error)
		execute_process(COMMAND "${OPENFPM_MOLTENVK_LLVM_CONFIG}" --ldflags
			OUTPUT_VARIABLE _openfpm_llvm_ldflags
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE _openfpm_llvm_ldflags_result
			ERROR_VARIABLE _openfpm_llvm_ldflags_error)
		execute_process(COMMAND "${OPENFPM_MOLTENVK_LLVM_CONFIG}" --libs
				core irreader support
			OUTPUT_VARIABLE _openfpm_llvm_libraries
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE _openfpm_llvm_libraries_result
			ERROR_VARIABLE _openfpm_llvm_libraries_error)
		if (NOT _openfpm_llvm_cxxflags_result EQUAL 0 OR
			NOT _openfpm_llvm_ldflags_result EQUAL 0 OR
			NOT _openfpm_llvm_libraries_result EQUAL 0)
			message(FATAL_ERROR
				"Unable to query the LLVM configuration required by the Metal ABI tool:\n"
				"${_openfpm_llvm_cxxflags_error}"
				"${_openfpm_llvm_ldflags_error}"
				"${_openfpm_llvm_libraries_error}")
		endif()
		separate_arguments(_openfpm_llvm_cxxflags NATIVE_COMMAND
			"${_openfpm_llvm_cxxflags}")
		separate_arguments(_openfpm_llvm_ldflags NATIVE_COMMAND
			"${_openfpm_llvm_ldflags}")
		separate_arguments(_openfpm_llvm_libraries NATIVE_COMMAND
			"${_openfpm_llvm_libraries}")
		find_library(_openfpm_zstd_library zstd
			HINTS /opt/homebrew/opt/zstd/lib /usr/local/lib)
		if (NOT _openfpm_zstd_library)
			message(FATAL_ERROR "LLVM requires libzstd to build the Metal ABI tool")
		endif()
		set(_openfpm_apple_clang /usr/bin/clang++)
		set(_openfpm_abi_tool_dir "${CMAKE_BINARY_DIR}/openfpm-tools")
		set(_openfpm_abi_tool
			"${_openfpm_abi_tool_dir}/openfpm-moltenvk-kernel-abi")
		add_custom_command(OUTPUT "${_openfpm_abi_tool}"
			COMMAND "${CMAKE_COMMAND}" -E make_directory
				"${_openfpm_abi_tool_dir}"
			COMMAND "${_openfpm_apple_clang}" ${_openfpm_llvm_cxxflags}
				"${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoltenVKKernelABI.cpp"
				${_openfpm_llvm_ldflags} ${_openfpm_llvm_libraries}
				"${_openfpm_zstd_library}" -lz -lcurses -lm
				-o "${_openfpm_abi_tool}"
			DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoltenVKKernelABI.cpp"
			COMMENT "Building the OpenFPM Metal kernel ABI normalizer"
			VERBATIM)
		add_custom_target(openfpm_moltenvk_abi_tool
			DEPENDS "${_openfpm_abi_tool}")
		set(OPENFPM_MOLTENVK_ABI_TOOL "${_openfpm_abi_tool}"
			CACHE FILEPATH "" FORCE)
	endif()
endif()

if (ALPAKA_ROOT)
	set(DEFINE_HAVE_ALPAKA "#define HAVE_ALPAKA")
endif()

if (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL")
	## We need BOOST_CONTEXT
	if (NOT Boost_CONTEXT_FOUND)
		message(FATAL_ERROR "Error to run CUDA on CPU (SEQUENTIAL backend) we need Boost::context installed, it seems is not available in your boost installation")
	endif()
endif()

if (CUDA_ON_BACKEND STREQUAL "OpenMP")
	## We need OpenMP
	if (NOT OPENMP_FOUND OR NOT Boost_CONTEXT_FOUND)
		message(FATAL_ERROR "Error to run CUDA on CPU (OpenMP backend) we need Boost::context and OpenMP installed, it seems is not available in your system")
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

add_subdirectory (src)

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
	set(DEFINE_GARBAGE_INJECTOR ${DEFINE_GARBAGE_INJECTOR} CACHE INTERNAL "")
	set(CMAKE_CXX_EXTENSIONS ${CMAKE_CXX_EXTENSIONS} CACHE INTERNAL "")
	set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} CACHE INTERNAL "")
	set(WARNING_SUPPRESSION_AND_OPTION_NVCC ${WARNING_SUPPRESSION_AND_OPTION_NVCC}  CACHE INTERNAL "")
	set(WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT ${WARNING_SUPPRESSION_AND_OPTION_NVCC_TEXT} CACHE INTERNAL "")
	set(DEFINE_CUDIFY_USE_HIP ${DEFINE_CUDIFY_USE_HIP} CACHE INTERNAL "")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} CACHE INTERNAL "")
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} CACHE INTERNAL "")
	set(DEFINE_HAVE_OPENMP ${DEFINE_HAVE_OPENMP} CACHE INTERNAL "")
	set(DEFINE_CUDIFY_USE_HIP ${DEFINE_CUDIFY_USE_HIP} CACHE INTERNAL "")
	set(DEFINE_CUDIFY_BACKEND ${DEFINE_CUDIFY_BACKEND} CACHE INTERNAL "")
	set(DEFINE_CUDA_GPU ${DEFINE_CUDA_GPU}  CACHE INTERNAL "")
	set(DEFINE_HAVE_OPENMP ${DEFINE_HAVE_OPENMP} CACHE INTERNAL "")
	set(DEFINE_HAVE_ALPAKA ${DEFINE_HAVE_ALPAKA} CACHE INTERNAL "")
	set(VULKAN_INCLUDE_DIR ${VULKAN_INCLUDE_DIR} CACHE INTERNAL "")
	set(MOLTENVK_LIBRARY ${MOLTENVK_LIBRARY} CACHE INTERNAL "")
	set(CUDA_FOUND ${CUDA_FOUND} CACHE INTERNAL "")
	set(CUDA_VERSION_MAJOR ${CUDA_VERSION_MAJOR} CACHE INTERNAL "")
	set(CUDA_VERSION_MINOR ${CUDA_VERSION_MINOR} CACHE INTERNAL "")
	set(CUDA_VERSION_PATCH ${CUDA_VERSION_PATCH} CACHE INTERNAL "")
endif()
