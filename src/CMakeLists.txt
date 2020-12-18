if(CUDA_FOUND OR CUDA_ON_CPU)
	set(CUDA_SOURCES memory/CudaMemory.cu)
else()
	set(CUDA_SOURCES )
endif()

if (CUDA_ON_CPU)
	add_definitions(-DCUDA_ON_CPU)
        set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
       	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        	add_definitions("-x c++")
        endif()
endif()

add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify_vars.cpp util/cudify_unit_test.cpp ${CUDA_SOURCES})

add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify_vars.cpp memory/PtrMemory.cpp ${CUDA_SOURCES})
add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify_vars.cpp memory/PtrMemory.cpp ${CUDA_SOURCES})

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
target_include_directories (mem PUBLIC ${ALPAKA_ROOT}/include)
if (CUDA_FOUND)
	target_include_directories (mem PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

target_include_directories (ofpmmemory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (ofpmmemory PUBLIC ${ALPAKA_ROOT}/include)

target_include_directories (ofpmmemory_dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory_dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
target_include_directories (ofpmmemory_dl PUBLIC ${ALPAKA_ROOT}/include)
target_link_libraries(mem ${Boost_LIBRARIES_BCK})

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


# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(mem PUBLIC cxx_std_11)


#if(BUILD_TESTING)

#  add_executable(particle_test test.cu)

#  set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#  target_link_libraries(particle_test PRIVATE particles)

#  add_test(NAME particles_10k COMMAND particle_test 10000 )
#  add_test(NAME particles_256k COMMAND particle_test 256000 )

#  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    #    set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #  endif()
    #endif()

