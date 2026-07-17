include_guard(GLOBAL)

include(CMakeParseArguments)

# Add an executable from OpenFPM's CUDA-style GPU sources.  The backend is an
# installation property: consumers keep using the original .cu files and do
# not need backend-specific targets or build rules.
function(openfpm_add_gpu_executable target)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(OPENFPM_GPU "${options}" "${one_value_args}"
        "${multi_value_args}" ${ARGN})

    if (NOT OPENFPM_GPU_SOURCES)
        message(FATAL_ERROR
            "openfpm_add_gpu_executable(${target}) requires SOURCES")
    endif()
    if (TARGET "${target}")
        message(FATAL_ERROR "Target ${target} already exists")
    endif()

    set(_openfpm_gpu_backend "${OPENFPM_CUDA_ON_BACKEND}")
    if (_openfpm_gpu_backend STREQUAL "CUDA")
        enable_language(CUDA)
        add_executable(${target} ${OPENFPM_GPU_SOURCES})
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS OFF)
    elseif (_openfpm_gpu_backend STREQUAL "HIP")
        find_package(HIP REQUIRED)
        hip_add_executable(${target} ${OPENFPM_GPU_SOURCES})
        target_compile_definitions(${target} PRIVATE __NVCC__ __HIP__
            CUDART_VERSION=11000 __CUDACC__ __CUDACC_VER_MAJOR__=11
            __CUDACC_VER_MINOR__=0 __CUDACC_VER_BUILD__=0)
    elseif (_openfpm_gpu_backend STREQUAL "METAL")
        if (NOT APPLE)
            message(FATAL_ERROR
                "The installed OpenFPM METAL backend requires macOS")
        endif()

        add_executable(${target} ${OPENFPM_GPU_SOURCES})
        target_compile_features(${target} PRIVATE cxx_std_17)

        set(_openfpm_gpu_includes)
        set(_openfpm_gpu_definitions __NVCC__ CUDIFY_USE_METAL CUDA_GPU)
        if (TARGET openfpm::binary_config_)
            get_target_property(_openfpm_gpu_target_includes
                openfpm::binary_config_ INTERFACE_INCLUDE_DIRECTORIES)
            get_target_property(_openfpm_gpu_target_definitions
                openfpm::binary_config_ INTERFACE_COMPILE_DEFINITIONS)
            if (_openfpm_gpu_target_includes)
                list(APPEND _openfpm_gpu_includes
                    ${_openfpm_gpu_target_includes})
            endif()
            if (_openfpm_gpu_target_definitions)
                list(APPEND _openfpm_gpu_definitions
                    ${_openfpm_gpu_target_definitions})
            endif()
        endif()
        # FindMPI can model wrapper-provided headers only on MPI::MPI_CXX;
        # pass those explicitly to the separate Metal HIP frontend.
        if (TARGET MPI::MPI_CXX)
            get_target_property(_openfpm_gpu_mpi_includes MPI::MPI_CXX
                INTERFACE_INCLUDE_DIRECTORIES)
            if (_openfpm_gpu_mpi_includes)
                list(APPEND _openfpm_gpu_includes
                    ${_openfpm_gpu_mpi_includes})
            endif()
        endif()
        list(REMOVE_DUPLICATES _openfpm_gpu_includes)
        list(REMOVE_DUPLICATES _openfpm_gpu_definitions)

        foreach(_openfpm_gpu_source IN LISTS OPENFPM_GPU_SOURCES)
            if (_openfpm_gpu_source MATCHES "\\.cu$")
                set_source_files_properties("${_openfpm_gpu_source}"
                    PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
                openfpm_add_moltenvk_kernels(
                    TARGET ${target}
                    SOURCE "${_openfpm_gpu_source}"
                    COMPILE_DEFINITIONS ${_openfpm_gpu_definitions}
                    INCLUDE_DIRECTORIES ${_openfpm_gpu_includes})
            endif()
        endforeach()
    elseif (_openfpm_gpu_backend STREQUAL "SEQUENTIAL" OR
            _openfpm_gpu_backend STREQUAL "OpenMP")
        add_executable(${target} ${OPENFPM_GPU_SOURCES})
        foreach(_openfpm_gpu_source IN LISTS OPENFPM_GPU_SOURCES)
            if (_openfpm_gpu_source MATCHES "\\.cu$")
                set_source_files_properties("${_openfpm_gpu_source}"
                    PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
            endif()
        endforeach()
        target_compile_definitions(${target} PRIVATE __NVCC__
            CUDART_VERSION=11000 __CUDACC_VER_MAJOR__=11
            __CUDACC_VER_MINOR__=0 __CUDACC_VER_BUILD__=0)
        if (_openfpm_gpu_backend STREQUAL "OpenMP")
            find_package(OpenMP REQUIRED)
            target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    elseif (_openfpm_gpu_backend STREQUAL "NONE" OR
            _openfpm_gpu_backend STREQUAL "None")
        # A no-GPU installation still accepts CUDA-style example sources.
        # Compile them as ordinary C++; their existing __NVCC__ guards select
        # the host-only/stub path without enabling a cudification runtime.
        add_executable(${target} ${OPENFPM_GPU_SOURCES})
        foreach(_openfpm_gpu_source IN LISTS OPENFPM_GPU_SOURCES)
            if (_openfpm_gpu_source MATCHES "\\.cu$")
                set_source_files_properties("${_openfpm_gpu_source}"
                    PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
            endif()
        endforeach()
    else()
        message(FATAL_ERROR
            "Unsupported installed OpenFPM GPU backend: ${_openfpm_gpu_backend}")
    endif()
endfunction()
