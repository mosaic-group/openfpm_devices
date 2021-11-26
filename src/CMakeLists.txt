if(NOT CUDA_ON_BACKEND STREQUAL "None")
	set(CUDA_SOURCES memory/CudaMemory.cu )
       	set(CUDA_SOURCES_TEST util/cudify/cudify_unit_test.cu)

	if (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL" OR CUDA_ON_BACKEND STREQUAL "OpenMP" OR CUDA_ON_BACKEND STREQUAL "HIP")
          set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
          set_source_files_properties(${CUDA_SOURCES_TEST} PROPERTIES LANGUAGE CXX)
	endif()

	if (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL" OR CUDA_ON_BACKEND STREQUAL "OpenMP" )
        	set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS "-D__NVCC__ -DCUDART_VERSION=11000")
    		if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        		add_definitions("-x c++")
        	endif()
	endif()
else()
	set(CUDA_SOURCES )
	set(CUDA_SOURCES_TEST )
endif()

if ( CUDA_ON_BACKEND STREQUAL "HIP" AND HIP_FOUND )

        hip_add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp  memory/mem_conf.cpp  ${CUDA_SOURCES})
        hip_add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})

        set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

	hip_add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify/cudify_vars.cpp  memory/mem_conf.cpp  ${CUDA_SOURCES} ${CUDA_SOURCES_TEST})

	list(APPEND HIP_HIPCC_FLAGS -D__NVCC__ -D__HIP__  -DCUDART_VERSION=11000 -D__CUDACC__ -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=0 -D__CUDACC_VER_BUILD__=0 --std=c++14)

	set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-fPIC")
        set_property(TARGET ofpmmemory PROPERTY NO_SONAME ON)
        set_property(TARGET ofpmmemory_dl PROPERTY NO_SONAME ON)
else()

	add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify/cudify_vars.cpp util/cudify/cudify_unit_test.cu memory/mem_conf.cpp  ${CUDA_SOURCES} ${CUDA_SOURCES_TEST})
	set_property(TARGET mem PROPERTY CUDA_ARCHITECTURES OFF)
	add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})
	set_property(TARGET ofpmmemory PROPERTY CUDA_ARCHITECTURES OFF)
	add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})
	set_property(TARGET ofpmmemory_dl PROPERTY CUDA_ARCHITECTURES OFF)
endif()

set_property(TARGET mem PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET mem PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ofpmmemory PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ofpmmemory_dl PROPERTY POSITION_INDEPENDENT_CODE ON)

if (HIP_FOUND)
        SET(CMAKE_EXE_LINKER_FLAGS  "--amdgpu-target=${AMD_ARCH_COMPILE}")
        SET(CMAKE_SHARED_LINKER_FLAGS  "--amdgpu-target=${AMD_ARCH_COMPILE}")
endif()


if ( CMAKE_COMPILER_IS_GNUCC )
    target_compile_options(mem PRIVATE "-Wno-deprecated-declarations")
    if (TEST_COVERAGE)
        target_compile_options(mem PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fprofile-arcs -ftest-coverage>)
    endif()
endif()


if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
	add_definitions(-D__STRICT_ANSI__)
endif()

if (CUDA_FOUND)
	target_include_directories(mem PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        	target_compile_options(mem PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe "--display_error_number --diag_suppress=2885 --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111" --expt-extended-lambda>)
	endif()
	if (TEST_COVERAGE)
                target_compile_options(mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "-fprofile-arcs -ftest-coverage" >)
    endif()

    if (CUDA_ON_BACKEND STREQUAL "CUDA")
        set_source_files_properties(memory/mem_conf.cpp PROPERTIES LANGUAGE CUDA)
    endif()

endif()

target_include_directories (mem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (mem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (mem PUBLIC ${Boost_INCLUDE_DIRS})
if (ALPAKA_ROOT)
	target_include_directories (mem PUBLIC ${ALPAKA_ROOT}/include)
endif()

target_include_directories (ofpmmemory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (ofpmmemory PUBLIC ${ALPAKA_ROOT}/include)
target_include_directories (ofpmmemory PUBLIC ${Boost_INCLUDE_DIRS})

target_include_directories (ofpmmemory_dl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory_dl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (ofpmmemory_dl PUBLIC ${ALPAKA_ROOT}/include)
target_include_directories (ofpmmemory_dl PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(mem ${Boost_LIBRARIES_BCK})
if (OPENMP_FOUND)
	target_link_libraries(mem OpenMP::OpenMP_CXX)
	target_link_libraries(ofpmmemory OpenMP::OpenMP_CXX)
	target_link_libraries(ofpmmemory_dl OpenMP::OpenMP_CXX)
endif()
target_link_libraries(ofpmmemory ${Boost_LIBRARIES_BCK})
target_link_libraries(ofpmmemory_dl ${Boost_LIBRARIES_BCK})

if (TEST_COVERAGE)
    target_link_libraries(mem -lgcov)
endif()


install(TARGETS ofpmmemory ofpmmemory_dl
	DESTINATION openfpm_devices/lib 
	COMPONENT OpenFPM)

install(FILES memory/ExtPreAlloc.hpp 
	      memory/BHeapMemory.hpp  
	      memory/HeapMemory.hpp 
	      memory/memory.hpp 
	      memory/PtrMemory.hpp
	      memory/CudaMemory.cuh
	      util/util_unit_tests.hpp
	      DESTINATION openfpm_devices/include/memory
	      COMPONENT OpenFPM)

install(FILES memory/ExtPreAlloc.hpp
              memory/BHeapMemory.hpp
              memory/HeapMemory.hpp
              memory/memory.hpp
              memory/PtrMemory.hpp
              memory/CudaMemory.cuh
	      memory/mem_conf.hpp
              DESTINATION openfpm_devices/include/memory
	      COMPONENT OpenFPM)

install(FILES util/print_stack.hpp
	      util/cuda_launch.hpp
	      util/se_util.hpp
	      util/cuda_util.hpp
	      util/cuda_kernel_error_checker.hpp
	DESTINATION openfpm_devices/include/util
	COMPONENT OpenFPM)

install(FILES util/cudify/cudify_alpaka.hpp
              util/cudify/cudify_hardware_alpaka.hpp
              util/cudify/cudify_hardware_common.hpp
              util/cudify/cudify_sequencial.hpp
	      util/cudify/cudify_openmp.hpp
	      util/cudify/cudify_hip.hpp
              util/cudify/cudify.hpp
	      util/cudify/cudify_cuda.hpp
        DESTINATION openfpm_devices/include/util/cudify
	COMPONENT OpenFPM)

install(FILES ptr_info.hpp
	DESTINATION openfpm_devices/include
	COMPONENT OpenFPM)


