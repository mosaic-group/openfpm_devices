if(CUDA_FOUND OR CUDA_ON_CPU)
	set(CUDA_SOURCES memory/CudaMemory.cu util/cudify/cudify_unit_test.cu)
else()
	set(CUDA_SOURCES )
endif()

if (CUDA_ON_CPU)
	add_definitions(-DCUDA_ON_CPU)
        set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
	set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS "-D__NVCC__ -DCUDA_GPU -DCUDART_VERSION=11000")
       	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        	add_definitions("-x c++")
        endif()
endif()

add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify/cudify_vars.cpp ${CUDA_SOURCES})

add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp ${CUDA_SOURCES})
add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp ${CUDA_SOURCES})

if ( CMAKE_COMPILER_IS_GNUCC )
    target_compile_options(mem PRIVATE "-Wno-deprecated-declarations")
    if (TEST_COVERAGE)
        target_compile_options(mem PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fprofile-arcs -ftest-coverage>)
    endif()
endif()


if (CUDA_FOUND)
	target_compile_options(mem PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe "--display_error_number --diag_suppress=2885 --diag_suppress=2887  --diag_suppress=2888 --diag_suppress=186 --diag_suppress=111" --expt-extended-lambda>)
	if (TEST_COVERAGE)
                target_compile_options(mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "-fprofile-arcs -ftest-coverage" >)
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
target_link_libraries(mem ${Boost_LIBRARIES_BCK})
target_link_libraries(ofpmmemory ${Boost_LIBRARIES_BCK})
target_link_libraries(ofpmmemory_dl ${Boost_LIBRARIES_BCK})

if (TEST_COVERAGE)
    target_link_libraries(mem -lgcov)
endif()

install(TARGETS ofpmmemory ofpmmemory_dl
	DESTINATION openfpm_devices/lib )

install(FILES memory/ExtPreAlloc.hpp 
	      memory/BHeapMemory.hpp  
	      memory/HeapMemory.hpp 
	      memory/memory.hpp 
	      memory/PtrMemory.hpp
	      memory/CudaMemory.cuh
	      util/util_unit_tests.hpp
	      DESTINATION openfpm_devices/include/memory)

install(FILES util/print_stack.hpp
	DESTINATION openfpm_devices/include/util)


install(FILES util/se_util.hpp
	      util/cuda_util.hpp
        DESTINATION openfpm_devices/include/util)

install(FILES ptr_info.hpp
	DESTINATION openfpm_devices/include)


