if(CUDA_FOUND OR CUDA_ON_CPU)
	set(CUDA_SOURCES memory/CudaMemory.cu)
	set(CUDA_SOURCES_TEST #util/cudify/cudify_unit_test.cu 
						  util/cudify/cudify_openmp_unit_test.cu
						  )
else()
	set(CUDA_SOURCES )
	set(CUDA_SOURCES_TEST )
endif()

if (CUDA_ON_CPU)
	add_definitions(-DCUDA_ON_CPU)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
	set_source_files_properties(${CUDA_SOURCES_TEST} PROPERTIES LANGUAGE CXX)
	set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS "-D__NVCC__ -DCUDART_VERSION=11000")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    	add_definitions("-x c++")
	endif()
endif()

add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify/cudify_vars.cpp ${CUDA_SOURCES_TEST} memory/mem_conf.cpp  ${CUDA_SOURCES})

set_property(TARGET mem PROPERTY CUDA_ARCHITECTURES OFF)

add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})
add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})

if(OpenMP_CXX_FOUND)
    target_compile_options(mem PUBLIC ${OpenMP_CXX_FLAGS})
endif()

if (CUDA_FOUND AND NOT CUDA_ON_CPU)
	add_library(ofpmmemory_cuda_on_cpu STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})
	add_library(ofpmmemory_dl_cuda_on_cpu SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})

	target_compile_options(ofpmmemory_cuda_on_cpu PUBLIC -DCUDA_ON_CPU)
	target_compile_options(ofpmmemory_dl_cuda_on_cpu PUBLIC -DCUDA_ON_CPU)
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
	if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        	target_compile_options(mem PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe "--display_error_number --diag_suppress=2885 --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111" --expt-extended-lambda>)
	endif()
	if (TEST_COVERAGE)
                target_compile_options(mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "-fprofile-arcs -ftest-coverage" >)
    endif()

    if (NOT CUDA_ON_CPU)
        set_source_files_properties(memory/mem_conf.cpp PROPERTIES LANGUAGE CUDA)
    endif()

endif()

target_include_directories (mem PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories (mem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (mem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (mem PUBLIC ${Boost_INCLUDE_DIRS})
if (ALPAKA_ROOT)
	target_include_directories (mem PUBLIC ${ALPAKA_ROOT}/include)
endif()
if (CUDA_FOUND)
	target_include_directories (mem PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

target_include_directories (ofpmmemory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (ofpmmemory PUBLIC ${ALPAKA_ROOT}/include)
target_include_directories (ofpmmemory PUBLIC ${Boost_INCLUDE_DIRS})

target_include_directories (ofpmmemory_dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory_dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (ofpmmemory_dl PUBLIC ${ALPAKA_ROOT}/include)
target_include_directories (ofpmmemory_dl PUBLIC ${Boost_INCLUDE_DIRS})

if (CUDA_FOUND AND NOT CUDA_ON_CPU)
	target_include_directories (ofpmmemory_cuda_on_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_include_directories (ofpmmemory_cuda_on_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
	target_include_directories (ofpmmemory_cuda_on_cpu PUBLIC ${ALPAKA_ROOT}/include)
	target_include_directories (ofpmmemory_cuda_on_cpu PUBLIC ${Boost_INCLUDE_DIRS})

	target_include_directories (ofpmmemory_dl_cuda_on_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_include_directories (ofpmmemory_dl_cuda_on_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
	target_include_directories (ofpmmemory_dl_cuda_on_cpu PUBLIC ${ALPAKA_ROOT}/include)
	target_include_directories (ofpmmemory_dl_cuda_on_cpu PUBLIC ${Boost_INCLUDE_DIRS})
endif()

target_link_libraries(mem ${Boost_LIBRARIES_BCK})
target_link_libraries(ofpmmemory ${Boost_LIBRARIES_BCK})
target_link_libraries(ofpmmemory_dl ${Boost_LIBRARIES_BCK})

if(OpenMP_CXX_FOUND)
    target_link_libraries(mem ${OpenMP_CXX_FLAGS})
endif()

if (TEST_COVERAGE)
    target_link_libraries(mem -lgcov)
endif()

if (CUDA_FOUND AND NOT CUDA_ON_CPU)
	set(ADDITIONAL_OPENFPM_LIBS ofpmmemory_cuda_on_cpu ofpmmemory_dl_cuda_on_cpu)
endif()

install(TARGETS ofpmmemory ofpmmemory_dl ${ADDITIONAL_OPENFPM_LIBS}
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
              util/cudify/cudify.hpp
        DESTINATION openfpm_devices/include/util/cudify
	COMPONENT OpenFPM)

install(FILES ptr_info.hpp
	DESTINATION openfpm_devices/include
	COMPONENT OpenFPM)


