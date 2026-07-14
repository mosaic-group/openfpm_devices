if (CUDA_ON_BACKEND STREQUAL "METAL")
    # CudaMemory.cuh selects MoltenVKMemory for this backend. Do not compile
    # the native CUDA allocator implementation into a Metal build.
    set(CUDA_SOURCES)
    set(CUDA_SOURCES_TEST)
    set(METAL_SOURCES
        memory/MoltenVKMemory.cpp
        util/metal/MoltenVKContext.cpp
        util/metal/MoltenVKKernel.cpp
        util/metal/MoltenVKHipRuntime.cpp
        util/metal/MoltenVKPrimitives.cpp)
    set(MEM_TEST_SOURCES memory/MoltenVKMemory_unit_test.cpp)
elseif(NOT CUDA_ON_BACKEND STREQUAL "None")
    set(CUDA_SOURCES memory/CudaMemory.cu )
    set(CUDA_SOURCES_TEST util/cudify/cudify_unit_test.cu)
    set(METAL_SOURCES)
    set(MEM_TEST_SOURCES ${CUDA_SOURCES_TEST})

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
    set(METAL_SOURCES)
    set(MEM_TEST_SOURCES util/cudify/cudify_unit_test.cu)
endif()

if ( CUDA_ON_BACKEND STREQUAL "HIP" AND HIP_FOUND )

    hip_add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp  memory/mem_conf.cpp  ${CUDA_SOURCES})
    hip_add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp  ${CUDA_SOURCES})

    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

    hip_add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify/cudify_vars.cpp  memory/mem_conf.cpp  ${CUDA_SOURCES} ${CUDA_SOURCES_TEST})

    list(APPEND HIP_HIPCC_FLAGS -D__NVCC__ -D__HIP__  -DCUDART_VERSION=11000 -D__CUDACC__ -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=0 -D__CUDACC_VER_BUILD__=0 --std=c++17)

    set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-fPIC")
        set_property(TARGET ofpmmemory PROPERTY NO_SONAME ON)
        set_property(TARGET ofpmmemory_dl PROPERTY NO_SONAME ON)
else()

    add_executable(mem main.cpp memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/mem_conf.cpp ${CUDA_SOURCES} ${METAL_SOURCES} ${MEM_TEST_SOURCES})
    set_property(TARGET mem PROPERTY CUDA_ARCHITECTURES OFF)
    add_library(ofpmmemory STATIC memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp ${CUDA_SOURCES} ${METAL_SOURCES})
    set_property(TARGET ofpmmemory PROPERTY CUDA_ARCHITECTURES OFF)
    add_library(ofpmmemory_dl SHARED memory/HeapMemory.cpp util/cudify/cudify_vars.cpp memory/PtrMemory.cpp memory/mem_conf.cpp ${CUDA_SOURCES} ${METAL_SOURCES})
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
target_include_directories (mem PUBLIC ${CMAKE_BINARY_DIR}/config)
target_include_directories (mem PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories (mem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/config)
if (ALPAKA_ROOT)
    target_include_directories (mem PUBLIC ${ALPAKA_ROOT}/include)
endif()

target_include_directories (ofpmmemory PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories (ofpmmemory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory PRIVATE ${CMAKE_BINARY_DIR}/config)
target_include_directories (ofpmmemory PUBLIC ${ALPAKA_ROOT}/include)
target_include_directories (ofpmmemory PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories (ofpmmemory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/config)

target_include_directories (ofpmmemory_dl PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories (ofpmmemory_dl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (ofpmmemory_dl PRIVATE ${CMAKE_BINARY_DIR}/config)
target_include_directories (ofpmmemory_dl PUBLIC ${ALPAKA_ROOT}/include)
target_include_directories (ofpmmemory_dl PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories (ofpmmemory_dl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/config)

if (CUDA_ON_BACKEND STREQUAL "METAL")
    foreach(target mem ofpmmemory ofpmmemory_dl)
        set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
        set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
        if (NOT CMAKE_PROJECT_NAME STREQUAL "openfpm")
            target_compile_definitions(${target} PUBLIC CUDIFY_USE_METAL CUDA_GPU)
        endif()
        target_include_directories(${target} PUBLIC ${VULKAN_INCLUDE_DIR})
        target_link_libraries(${target} ${MOLTENVK_LIBRARY}
            "-framework Foundation" "-framework QuartzCore" "-framework Metal"
            "-framework IOSurface")
    endforeach()

    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenFPMMoltenVK.cmake")
    foreach(target mem ofpmmemory ofpmmemory_dl)
        openfpm_add_moltenvk_kernels(
            TARGET ${target}
            SOURCE util/metal/MoltenVKPrimitives.cu
            HIP_CLANG "${OPENFPM_MOLTENVK_HIP_CLANG}"
            LLVM_OPT "${OPENFPM_MOLTENVK_LLVM_OPT}"
            CLSPV "${OPENFPM_MOLTENVK_CLSPV}"
            CHIPSTAR_ROOT "${OPENFPM_CHIPSTAR_ROOT}"
            CHIPSTAR_GENERATED_INCLUDE "${OPENFPM_CHIPSTAR_GENERATED_INCLUDE}"
            REGISTRATION_FUNCTION openfpm_register_moltenvk_primitives
            COMPILE_DEFINITIONS CUDIFY_USE_METAL CUDA_GPU
            INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_BINARY_DIR}/config" ${Boost_INCLUDE_DIRS})
    endforeach()
elseif (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND NOT CUDA_ON_BACKEND STREQUAL "None")
    # A parent OpenFPM build writes these selections to its generated
    # config.h. Standalone openfpm_devices builds need equivalent definitions.
    foreach(target mem ofpmmemory ofpmmemory_dl)
        target_compile_definitions(${target} PRIVATE CUDA_GPU)
        if (CUDA_ON_BACKEND STREQUAL "SEQUENTIAL")
            target_compile_definitions(${target} PRIVATE CUDIFY_USE_SEQUENTIAL)
        elseif (CUDA_ON_BACKEND STREQUAL "OpenMP")
            target_compile_definitions(${target} PRIVATE CUDIFY_USE_OPENMP)
        elseif (CUDA_ON_BACKEND STREQUAL "CUDA")
            target_compile_definitions(${target} PRIVATE CUDIFY_USE_CUDA)
        elseif (CUDA_ON_BACKEND STREQUAL "HIP")
            target_compile_definitions(${target} PRIVATE CUDIFY_USE_HIP)
        endif()
    endforeach()
endif()

target_link_libraries(mem ${Boost_LIBRARIES})
if (OPENMP_FOUND)
    target_link_libraries(mem OpenMP::OpenMP_CXX)
    target_link_libraries(ofpmmemory OpenMP::OpenMP_CXX)
    target_link_libraries(ofpmmemory_dl OpenMP::OpenMP_CXX)
endif()

target_link_libraries(ofpmmemory ${Boost_LIBRARIES})
target_link_libraries(ofpmmemory_dl ${Boost_LIBRARIES})

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_test(NAME openfpm_devices_memory COMMAND mem)
endif()

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

install(FILES util/ofp_context.hpp
    util/gpu_context.hpp
    util/gpu_types.hpp
    util/print_stack.hpp
    util/se_util.hpp
    util/cuda_util.hpp
    util/cuda_kernel_error_checker.hpp
    DESTINATION openfpm_devices/include/util
    COMPONENT OpenFPM)

install(FILES util/cudify/alpaka/cudify_alpaka.hpp
    util/cudify/alpaka/cudify_hardware_alpaka.hpp
    DESTINATION openfpm_devices/include/util/cudify/alpaka
    COMPONENT OpenFPM)

install(FILES util/cudify/sequential/cudify_sequential.hpp
    DESTINATION openfpm_devices/include/util/cudify/sequential
    COMPONENT OpenFPM)

install(FILES util/cudify/openmp/cudify_openmp.hpp
    DESTINATION openfpm_devices/include/util/cudify/openmp
    COMPONENT OpenFPM)

install(FILES util/cudify/hip/cudify_hip.hpp
    DESTINATION openfpm_devices/include/util/cudify/hip
    COMPONENT OpenFPM)

install(FILES util/cudify/cuda/cudify_cuda.hpp
    util/cudify/cuda/operators.hpp
    DESTINATION openfpm_devices/include/util/cudify/cuda
    COMPONENT OpenFPM)

install(FILES util/cudify/metal/cudify_metal.hpp
    DESTINATION openfpm_devices/include/util/cudify/metal
    COMPONENT OpenFPM)

install(FILES util/metal/MoltenVKContext.hpp
    util/metal/MoltenVKKernel.hpp
    util/metal/MoltenVKPrimitives.hpp
    DESTINATION openfpm_devices/include/util/metal
    COMPONENT OpenFPM)

install(DIRECTORY util/metal/compat/include/
    DESTINATION openfpm_devices/include/openfpm_moltenvk_compat
    COMPONENT OpenFPM)

install(FILES memory/MoltenVKMemory.hpp
    DESTINATION openfpm_devices/include/memory
    COMPONENT OpenFPM)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenFPMMoltenVK.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompileMoltenVKKernels.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/MoltenVKKernelABI.cpp
    DESTINATION openfpm_devices/cmake
    COMPONENT OpenFPM)

if (CUDA_ON_BACKEND STREQUAL "METAL")
    install(PROGRAMS "${OPENFPM_MOLTENVK_ABI_TOOL}"
        DESTINATION openfpm_devices/bin
        COMPONENT OpenFPM)
endif()

install(FILES util/cudify/cudify_hardware_cpu.hpp
    DESTINATION openfpm_devices/include/util/cudify
    COMPONENT OpenFPM)
