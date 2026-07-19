include(CMakeParseArguments)

# Capture the helper location at include time. This is equivalent to
# CMAKE_CURRENT_FUNCTION_LIST_DIR but remains compatible with CMake 3.8.
set(_OPENFPM_MOLTENVK_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Compile the device half of a HIP/CUDA-style translation unit to Vulkan
# SPIR-V and add a generated registrar to TARGET.  When SOURCE already belongs
# to TARGET, HIP clang also compiles its host half so compiler-emitted kernel
# handles select exact template specializations at runtime.
function(openfpm_add_moltenvk_kernels)
    set(options)
    set(one_value_args TARGET SOURCE HIP_CLANG LLVM_OPT LLVM_LINK CLSPV
        SPIRV_VAL ABI_TOOL CHIPSTAR_ROOT CHIPSTAR_GENERATED_INCLUDE
        REGISTRATION_FUNCTION)
    set(multi_value_args INCLUDE_DIRECTORIES COMPILE_DEFINITIONS)
    cmake_parse_arguments(MVK "${options}" "${one_value_args}"
        "${multi_value_args}" ${ARGN})

    if (NOT MVK_HIP_CLANG)
        set(MVK_HIP_CLANG "${OPENFPM_MOLTENVK_HIP_CLANG}")
    endif()
    if (NOT MVK_LLVM_OPT)
        set(MVK_LLVM_OPT "${OPENFPM_MOLTENVK_LLVM_OPT}")
    endif()
    if (NOT MVK_LLVM_LINK)
        set(MVK_LLVM_LINK "${OPENFPM_MOLTENVK_LLVM_LINK}")
    endif()
    if (NOT MVK_LLVM_LINK AND MVK_LLVM_OPT)
        get_filename_component(_openfpm_mvk_llvm_bin "${MVK_LLVM_OPT}"
            DIRECTORY)
        if (EXISTS "${_openfpm_mvk_llvm_bin}/llvm-link")
            set(MVK_LLVM_LINK "${_openfpm_mvk_llvm_bin}/llvm-link")
        endif()
    endif()
    if (NOT MVK_ABI_TOOL)
        set(MVK_ABI_TOOL "${OPENFPM_MOLTENVK_ABI_TOOL}")
    endif()
	if (NOT MVK_CLSPV)
		set(MVK_CLSPV "${OPENFPM_MOLTENVK_CLSPV}")
	endif()
	if (NOT MVK_CHIPSTAR_ROOT)
		set(MVK_CHIPSTAR_ROOT "${OPENFPM_CHIPSTAR_ROOT}")
	endif()
	if (NOT MVK_CHIPSTAR_GENERATED_INCLUDE)
		set(MVK_CHIPSTAR_GENERATED_INCLUDE
			"${OPENFPM_CHIPSTAR_GENERATED_INCLUDE}")
	endif()
	if (NOT MVK_SPIRV_VAL)
		set(MVK_SPIRV_VAL "${OPENFPM_MOLTENVK_SPIRV_VAL}")
	endif()
	if (NOT MVK_SPIRV_VAL AND MVK_CLSPV)
		get_filename_component(_openfpm_mvk_clspv_bin "${MVK_CLSPV}"
			DIRECTORY)
		find_program(_openfpm_mvk_spirv_val NAMES spirv-val
			HINTS "${_openfpm_mvk_clspv_bin}")
		if (_openfpm_mvk_spirv_val)
			set(MVK_SPIRV_VAL "${_openfpm_mvk_spirv_val}")
		endif()
	endif()
    if (NOT MVK_ABI_TOOL)
        find_program(_openfpm_mvk_installed_abi_tool
            NAMES openfpm-moltenvk-kernel-abi
            HINTS "${_OPENFPM_MOLTENVK_CMAKE_DIR}/../bin"
            NO_DEFAULT_PATH)
        if (_openfpm_mvk_installed_abi_tool)
            set(MVK_ABI_TOOL "${_openfpm_mvk_installed_abi_tool}")
        endif()
    endif()

    foreach(required TARGET SOURCE HIP_CLANG LLVM_OPT LLVM_LINK CLSPV SPIRV_VAL
			ABI_TOOL CHIPSTAR_ROOT)
        if (NOT MVK_${required})
            message(FATAL_ERROR "openfpm_add_moltenvk_kernels requires ${required}")
        endif()
    endforeach()
    if (NOT TARGET ${MVK_TARGET})
        message(FATAL_ERROR "Unknown MoltenVK kernel target: ${MVK_TARGET}")
    endif()

    # Keep the shared OpenFPM sources on their ordinary HIP includes.  Only
    # translated Metal TUs see these narrow hipCUB/Thrust compatibility
    # headers, so native CUDA and HIP include resolution is never shadowed.
    set(_openfpm_mvk_compat_include)
    foreach(_openfpm_mvk_compat_candidate
        "${_OPENFPM_MOLTENVK_CMAKE_DIR}/../src/util/metal/compat/include"
        "${_OPENFPM_MOLTENVK_CMAKE_DIR}/../include/openfpm_moltenvk_compat")
        if (EXISTS "${_openfpm_mvk_compat_candidate}/hipcub/hipcub.hpp")
            set(_openfpm_mvk_compat_include
                "${_openfpm_mvk_compat_candidate}")
            break()
        endif()
    endforeach()
    if (NOT _openfpm_mvk_compat_include)
        message(FATAL_ERROR
            "The OpenFPM MoltenVK hipCUB/Thrust compatibility headers are missing")
    endif()
    set(_openfpm_mvk_frontend_compat
        "${_openfpm_mvk_compat_include}/openfpm/metal/HipCudaFrontendCompat.hpp")
    if (NOT EXISTS "${_openfpm_mvk_frontend_compat}")
        message(FATAL_ERROR
            "The OpenFPM Metal HIP/CUDA frontend compatibility header is missing")
    endif()

    get_filename_component(source "${MVK_SOURCE}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    get_filename_component(stem "${source}" NAME_WE)
    set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/moltenvk/${MVK_TARGET}_${stem}")
    file(MAKE_DIRECTORY "${out_dir}")
    set(flags_file "${out_dir}/device_flags.rsp")
    set(host_flags_file "${out_dir}/host_flags.rsp")
    set(depfile "${out_dir}/${stem}.d")
    set(generated_cpp "${out_dir}/${stem}_spirv.cpp")
    set(host_object "${out_dir}/${stem}_host.o")

    set(compile_host FALSE)
    get_target_property(target_source_list ${MVK_TARGET} SOURCES)
    foreach(target_source IN LISTS target_source_list)
        if (target_source MATCHES "^\\$<")
            continue()
        endif()
        get_filename_component(target_source_absolute "${target_source}"
            ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if (target_source_absolute STREQUAL source)
            set(compile_host TRUE)
        endif()
    endforeach()

    execute_process(COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE macos_sdk OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE sdk_result)
    if (NOT sdk_result EQUAL 0)
        message(FATAL_ERROR "Unable to locate the macOS SDK with xcrun")
    endif()

    set(common_flags
        "-x\nhip\n--offload=spirv64\n--no-offload-new-driver\n-nohipwrapperinc\n-no-hip-rt\n-nogpulib\n--hip-path=${MVK_CHIPSTAR_ROOT}\n--target=arm64-apple-darwin\n-O2\n-D__HIP_PLATFORM_SPIRV__\n-DNDEBUG\n--sysroot=${macos_sdk}\n-I${MVK_CHIPSTAR_ROOT}/HIP/include\n-I${MVK_CHIPSTAR_ROOT}/include\n-I${MVK_CHIPSTAR_ROOT}/include/cuspv\n-include\n${_openfpm_mvk_frontend_compat}\n")
	file(WRITE "${flags_file}" "${common_flags}"
		"--cuda-device-only\n-emit-llvm\n-c\n-Xarch_device\n"
		"-I${MVK_CHIPSTAR_ROOT}/include/hip/devicelib/macOS\n"
		"-I${_openfpm_mvk_compat_include}\n")
    file(WRITE "${host_flags_file}" "${common_flags}"
		"--cuda-host-only\n-fPIC\n-std=c++17\n"
        "-I${_openfpm_mvk_compat_include}\n")
    if (MVK_CHIPSTAR_GENERATED_INCLUDE)
        file(APPEND "${flags_file}" "-I${MVK_CHIPSTAR_GENERATED_INCLUDE}\n")
        file(APPEND "${host_flags_file}"
            "-I${MVK_CHIPSTAR_GENERATED_INCLUDE}\n")
    endif()
    foreach(definition IN LISTS MVK_COMPILE_DEFINITIONS)
        file(APPEND "${flags_file}" "-D${definition}\n")
        file(APPEND "${host_flags_file}" "-D${definition}\n")
    endforeach()
    foreach(include_dir IN LISTS MVK_INCLUDE_DIRECTORIES)
        get_filename_component(include_dir "${include_dir}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        file(APPEND "${flags_file}" "-I${include_dir}\n")
        file(APPEND "${host_flags_file}" "-I${include_dir}\n")
    endforeach()

    set(depfile_arguments)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.20 OR
        CMAKE_GENERATOR MATCHES "Ninja")
        list(APPEND depfile_arguments DEPFILE "${depfile}")
    endif()

    set(command_outputs "${generated_cpp}")
    if (compile_host)
        list(APPEND command_outputs "${host_object}")
    else()
        set(host_object)
    endif()
    set(abi_dependency "${MVK_ABI_TOOL}")
    if (TARGET openfpm_moltenvk_abi_tool)
        list(APPEND abi_dependency openfpm_moltenvk_abi_tool)
    endif()

    add_custom_command(
        OUTPUT ${command_outputs}
        COMMAND "${CMAKE_COMMAND}"
            "-DSOURCE=${source}"
            "-DOUTPUT_DIR=${out_dir}"
            "-DOUTPUT_CPP=${generated_cpp}"
            "-DFLAGS_FILE=${flags_file}"
            "-DHOST_FLAGS_FILE=${host_flags_file}"
            "-DHOST_OUTPUT=${host_object}"
            "-DDEPFILE=${depfile}"
            "-DHIP_CLANG=${MVK_HIP_CLANG}"
            "-DLLVM_OPT=${MVK_LLVM_OPT}"
            "-DLLVM_LINK=${MVK_LLVM_LINK}"
            "-DCLSPV=${MVK_CLSPV}"
			"-DSPIRV_VAL=${MVK_SPIRV_VAL}"
            "-DABI_TOOL=${MVK_ABI_TOOL}"
            "-DCHIPSTAR_ROOT=${MVK_CHIPSTAR_ROOT}"
            "-DREGISTRATION_FUNCTION=${MVK_REGISTRATION_FUNCTION}"
            -P "${_OPENFPM_MOLTENVK_CMAKE_DIR}/CompileMoltenVKKernels.cmake"
        DEPENDS "${source}" "${flags_file}" "${host_flags_file}"
            ${abi_dependency}
            "${_openfpm_mvk_frontend_compat}"
            "${MVK_CHIPSTAR_ROOT}/bitcode/devicelib.cl"
            "${_OPENFPM_MOLTENVK_CMAKE_DIR}/CompileMoltenVKKernels.cmake"
        BYPRODUCTS "${out_dir}/${stem}.bc"
            "${out_dir}/${stem}.linked.bc" "${out_dir}/${stem}.ll"
            "${out_dir}/chipstar-devicelib.bc"
        ${depfile_arguments}
        COMMENT "Translating ${stem} HIP/CUDA kernels to MoltenVK SPIR-V"
        VERBATIM)
    set_source_files_properties("${generated_cpp}" PROPERTIES GENERATED TRUE)
    target_sources(${MVK_TARGET} PRIVATE "${generated_cpp}")
    if (compile_host)
        set_source_files_properties("${source}" PROPERTIES HEADER_FILE_ONLY TRUE)
        set_source_files_properties("${host_object}" PROPERTIES
            GENERATED TRUE EXTERNAL_OBJECT TRUE)
        target_sources(${MVK_TARGET} PRIVATE "${host_object}")
    endif()
endfunction()
