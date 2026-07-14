if (NOT SOURCE OR NOT OUTPUT_DIR OR NOT OUTPUT_CPP OR NOT FLAGS_FILE OR
	NOT HOST_FLAGS_FILE OR NOT HIP_CLANG OR NOT LLVM_OPT OR NOT LLVM_LINK OR
	NOT CLSPV OR NOT SPIRV_VAL OR NOT ABI_TOOL OR NOT CHIPSTAR_ROOT)
	message(FATAL_ERROR "Incomplete MoltenVK kernel compiler invocation")
endif()

get_filename_component(stem "${SOURCE}" NAME_WE)
find_program(HEXDUMP_EXECUTABLE NAMES hexdump)
if (NOT HEXDUMP_EXECUTABLE)
	message(FATAL_ERROR "hexdump is required to embed generated Metal SPIR-V")
endif()
# Remove outputs from the former one-module-per-translation-unit pipeline so
# incremental build trees cannot mistake them for current embedded modules.
file(REMOVE "${OUTPUT_DIR}/${stem}.spv" "${OUTPUT_DIR}/${stem}.raw.spv"
	"${OUTPUT_DIR}/${stem}.internal.ll" "${OUTPUT_DIR}/${stem}.clspv.ll")
set(bitcode "${OUTPUT_DIR}/${stem}.bc")
set(device_library "${OUTPUT_DIR}/chipstar-devicelib.bc")
set(linked_bitcode "${OUTPUT_DIR}/${stem}.linked.bc")
set(ir "${OUTPUT_DIR}/${stem}.ll")

function(run_checked description)
	execute_process(COMMAND ${ARGN} RESULT_VARIABLE result
		OUTPUT_VARIABLE output ERROR_VARIABLE error)
	if (NOT result EQUAL 0)
		message(FATAL_ERROR "${description} failed (${result})\n${output}${error}")
	endif()
endfunction()

run_checked("HIP dependency scan" "${HIP_CLANG}" "@${HOST_FLAGS_FILE}"
	-M -MT "${OUTPUT_CPP}" -MF "${DEPFILE}" "${SOURCE}")
run_checked("HIP device compilation" "${HIP_CLANG}" "@${FLAGS_FILE}"
	"${SOURCE}" -o "${bitcode}")
run_checked("chipStar device-library compilation" "${HIP_CLANG}"
	-Xclang -finclude-default-header -O2 -x cl -cl-std=CL2.0
	-cl-ext=+cl_khr_subgroup -DDEFAULT_WARP_SIZE=32 -emit-llvm -cl-no-stdinc
	--target=spirv64 -o "${device_library}" -c
	"${CHIPSTAR_ROOT}/bitcode/devicelib.cl")
run_checked("HIP device-library link" "${LLVM_LINK}" -o "${linked_bitcode}"
	"${bitcode}" "${device_library}")
run_checked("LLVM IR emission" "${LLVM_OPT}" -S "${linked_bitcode}" -o "${ir}")

file(READ "${ir}" ir_text)
string(REGEX MATCHALL "define[^\n]*spir_kernel[^\n]*@[^ (]+" kernel_lines
	"${ir_text}")
set(entries)
foreach(line IN LISTS kernel_lines)
	string(REGEX REPLACE ".*@([^ (]+).*" "\\1" entry "${line}")
	list(APPEND entries "${entry}")
endforeach()
list(REMOVE_DUPLICATES entries)
if (NOT entries)
	if (NOT HOST_OUTPUT)
		message(FATAL_ERROR
			"No instantiated __global__ kernels found in ${SOURCE}; "
			"a host object was not requested")
	endif()
	# Some historical .cu test translation units contain host-side GPU setup
	# without instantiating a __global__ function themselves. They still need
	# HIP clang's host pass so their ordinary C++ tests are linked into the
	# parent executable, but there is no SPIR-V module to register. HIP clang
	# nevertheless emits a TU-specific __hip_fatbin symbol reference in that
	# host object. Embed the already-produced device bitcode solely as an opaque
	# payload so clang defines the symbol; MoltenVKHipRuntime intentionally
	# ignores fatbin contents because executable SPIR-V is registered by the
	# generated C++ registrar.
	if (REGISTRATION_FUNCTION)
		file(WRITE "${OUTPUT_CPP}"
			"extern \"C\" void ${REGISTRATION_FUNCTION}() {}\n")
	else()
		file(WRITE "${OUTPUT_CPP}"
			"// No instantiated HIP/CUDA kernels in ${stem}.\n")
	endif()
	message(STATUS
		"No instantiated __global__ kernels in ${SOURCE}; compiling host object only")
	run_checked("HIP host compilation" "${HIP_CLANG}" "@${HOST_FLAGS_FILE}"
		-Xclang -fcuda-include-gpubinary -Xclang "${bitcode}"
		"${SOURCE}" -c -o "${HOST_OUTPUT}")
	return()
endif()

get_filename_component(clspv_directory "${CLSPV}" DIRECTORY)
set(clspv_reflection "${clspv_directory}/clspv-reflection")
set(module_definitions)
set(registration_statements)
set(host_embedded_binaries)
set(index 0)

foreach(entry IN LISTS entries)
	set(internal_ir "${OUTPUT_DIR}/${stem}.${index}.internal.ll")
	set(clspv_ir "${OUTPUT_DIR}/${stem}.${index}.clspv.ll")
	set(abi_ir "${OUTPUT_DIR}/${stem}.${index}.abi.ll")
	set(lowered_abi_ir "${OUTPUT_DIR}/${stem}.${index}.abi.opt.ll")
	set(compatible_abi_ir "${OUTPUT_DIR}/${stem}.${index}.abi.compat.ll")
	set(final_abi_ir "${OUTPUT_DIR}/${stem}.${index}.abi.final.ll")
	set(final_compatible_abi_ir
		"${OUTPUT_DIR}/${stem}.${index}.abi.final.compat.ll")
	set(abi_metadata "${OUTPUT_DIR}/${stem}.${index}.abi.json")
	set(push_constant_metadata
		"${OUTPUT_DIR}/${stem}.${index}.push_constants.json")
	set(raw_spirv "${OUTPUT_DIR}/${stem}.${index}.raw.spv")
	set(spirv "${OUTPUT_DIR}/${stem}.${index}.spv")

	# A Vulkan shader module has no linker.  Keep one exact kernel entry and
	# everything reachable from it, including the linked HIP device library.
	run_checked("LLVM kernel internalization for ${entry}" "${LLVM_OPT}" -S
		"${ir}" -passes=internalize,globaldce
		"-internalize-public-api-list=${entry}" -o "${internal_ir}")

	file(READ "${internal_ir}" cleaned)
	string(REPLACE "target triple = \"spirv64\""
		"target triple = \"spirv64-unknown-vulkan\"" cleaned "${cleaned}")
	string(REGEX REPLACE "[^\n]*@__chipspv_abort_called[^\n]*\n" "" cleaned
		"${cleaned}")
	string(REGEX REPLACE "[^\n]*@__hip_cuid_[^\n]*\n" "" cleaned "${cleaned}")
	string(REGEX REPLACE "[^\n]*@llvm.compiler.used[^\n]*\n" "" cleaned
		"${cleaned}")
	file(WRITE "${clspv_ir}" "${cleaned}")

	# Normalize arbitrary HIP/CUDA parameter lists to one packed record reached
	# through a physical-storage-buffer device address.
	# The sidecar is the authoritative host/device layout contract.
	run_checked("Metal kernel ABI normalization for ${entry}" "${ABI_TOOL}"
		"${clspv_ir}" "${abi_ir}" "${abi_metadata}")
	file(READ "${abi_metadata}" metadata)
	string(REGEX MATCH "\"entry\":\"([^\"]+)\"" ignored "${metadata}")
	set(metadata_entry "${CMAKE_MATCH_1}")
	if (NOT metadata_entry STREQUAL entry)
		message(FATAL_ERROR "Metal ABI metadata entry mismatch for ${entry}")
	endif()
	string(REGEX MATCH "\"size\":([0-9]+)" ignored "${metadata}")
	set(packed_size "${CMAKE_MATCH_1}")
	string(REGEX MATCH "\"alignment\":([0-9]+)" ignored "${metadata}")
	set(packed_alignment "${CMAKE_MATCH_1}")
	string(REGEX MATCHALL
		"\\{\"index\":[0-9]+,\"offset\":[0-9]+,\"size\":[0-9]+,\"alignment\":[0-9]+\\}"
		argument_records "${metadata}")

	set(layout_literals)
	set(argument_count 0)
	foreach(record IN LISTS argument_records)
		string(REGEX MATCH "\"offset\":([0-9]+)" ignored "${record}")
		set(argument_offset "${CMAKE_MATCH_1}")
		string(REGEX MATCH "\"size\":([0-9]+)" ignored "${record}")
		set(argument_size "${CMAKE_MATCH_1}")
		string(REGEX MATCH "\"alignment\":([0-9]+)" ignored "${record}")
		set(argument_alignment "${CMAKE_MATCH_1}")
		string(APPEND layout_literals
			"    {${argument_offset}, ${argument_size}, ${argument_alignment}},\n")
		math(EXPR argument_count "${argument_count} + 1")
	endforeach()
	if (argument_count EQUAL 0)
		set(layout_literals "    {0, 0, 1},\n")
	endif()

	# The ABI wrapper value-copies HIP byval aggregates from the physical
	# argument block into private storage. Inline the original kernel body and
	# scalarize those copies before clspv performs address-space inference.
	# Leaving this to clspv's late inliner triggers invalid AS1/private pointer
	# reconstruction for aggregate kernels.
	run_checked("LLVM ABI lowering for ${entry}" "${LLVM_OPT}" -S "${abi_ir}"
		"-passes=cgscc(inline),sroa,simplifycfg"
		-o "${lowered_abi_ir}")
	# HIP by-value aggregates can contain dynamically indexed arrays of device
	# pointers.  After inlining/SROA exposes their leaf accesses, store those
	# pointers as integer addresses so clspv does not have to model illegal
	# pointer-valued Vulkan private memory.
	run_checked("Metal LLVM compatibility for ${entry}" "${ABI_TOOL}"
		--llvm-compat "${lowered_abi_ir}" "${compatible_abi_ir}")
	# Core HIP kernels pass large nested C++ views by value. Fold the generated
	# aggregate reconstruction into direct scalar loads before clspv; this also
	# removes residual generic-pointer insertvalue chains that otherwise make its
	# LowerAddrSpaceCast pass fail to converge. The explicit primitive catalogue
	# already uses a compact ABI record, and this clspv revision miscompiles its
	# segmented-reduction initial scalar into a pointer phi after instcombine, so
	# preserve that translation unit's already-scalarized control flow.
	set(clspv_input_ir "${compatible_abi_ir}")
	if (NOT stem STREQUAL "MoltenVKPrimitives")
		run_checked("Metal LLVM aggregate cleanup for ${entry}"
			"${LLVM_OPT}" -S "${compatible_abi_ir}"
			-passes=instcombine,simplifycfg -o "${final_abi_ir}")
		run_checked("Metal post-cleanup LLVM compatibility for ${entry}"
			"${ABI_TOOL}" --llvm-compat "${final_abi_ir}"
			"${final_compatible_abi_ir}")
		set(clspv_input_ir "${final_compatible_abi_ir}")
	endif()
	run_checked("clspv SPIR-V lowering for ${entry}" "${CLSPV}" "${clspv_input_ir}"
		-x=ir -o "${raw_spirv}" -arch=spirv64 -cl-std=CL2.0
		-inline-entry-points -physical-storage-buffers)
	# Preserve ordinary HIP/CUDA kernels and repair the old SPIRV-Cross built-in
	# activity analysis shipped by MoltenVK 1.4.1 at the ABI boundary.
	run_checked("MoltenVK SPIR-V compatibility for ${entry}" "${ABI_TOOL}"
		--spirv-compat "${raw_spirv}" "${spirv}")
	# Validate the exact Vulkan 1.2 contract enabled by MoltenVKContext. Do not
	# hide layout defects behind optional validator feature flags.
	run_checked("Vulkan SPIR-V validation for ${entry}" "${SPIRV_VAL}"
		--target-env vulkan1.2 "${spirv}")
	run_checked("clspv push-constant reflection for ${entry}" "${ABI_TOOL}"
		--spirv-reflect "${spirv}" "${push_constant_metadata}")
	file(READ "${push_constant_metadata}" push_constant_json)
	foreach(push_field IN ITEMS argument_pointer global_offset
		enqueued_local_size global_size region_offset num_workgroups
		region_group_offset size)
		string(REGEX MATCH "\"${push_field}\":(-?[0-9]+)" ignored
			"${push_constant_json}")
		set(push_value "${CMAKE_MATCH_1}")
		if (ignored STREQUAL "")
			message(FATAL_ERROR
				"Missing ${push_field} in clspv reflection for ${entry}")
		endif()
		set(push_${push_field} "${push_value}")
	endforeach()

	if (EXISTS "${clspv_reflection}")
		run_checked("SPIR-V ABI reflection for ${entry}" "${clspv_reflection}"
			--target-env vulkan1.2 -d "${spirv}")
		execute_process(COMMAND "${clspv_reflection}" --target-env vulkan1.2
			-d "${spirv}" OUTPUT_VARIABLE reflection RESULT_VARIABLE reflection_result)
		if (NOT reflection_result EQUAL 0 OR
			NOT reflection MATCHES "kernel_decl,${entry}")
			message(FATAL_ERROR
				"SPIR-V reflection is missing kernel ${entry}\n${reflection}")
		endif()
	endif()

	file(SIZE "${spirv}" spirv_size)
	math(EXPR remainder "${spirv_size} % 4")
	if (NOT remainder EQUAL 0)
		message(FATAL_ERROR "Generated SPIR-V is not a sequence of 32-bit words")
	endif()
	# CMake's per-word string loop becomes quadratic for translation units with
	# hundreds of kernels.  Let the platform byte dumper format native
	# little-endian uint32 words in one process instead (Metal is macOS/arm64).
	execute_process(COMMAND "${HEXDUMP_EXECUTABLE}" -v -e
		"1/4 \"0x%08x,\\n\"" "${spirv}"
		OUTPUT_VARIABLE words RESULT_VARIABLE hexdump_result
		ERROR_VARIABLE hexdump_error)
	if (NOT hexdump_result EQUAL 0)
		message(FATAL_ERROR
			"SPIR-V embedding failed for ${entry}: ${hexdump_error}")
	endif()

	string(APPEND module_definitions
		"alignas(4) const std::uint32_t module_words_${index}[] = {\n${words}};\n"
		"const openfpm::metal::KernelArgumentLayout module_layout_${index}[] = {\n${layout_literals}};\n"
		"constexpr openfpm::metal::KernelPushConstantLayout module_push_${index}{\n"
		"    ${push_size}, ${push_argument_pointer}, ${push_global_offset},\n"
		"    ${push_enqueued_local_size}, ${push_global_size}, ${push_region_offset},\n"
		"    ${push_num_workgroups}, ${push_region_group_offset}};\n")
	string(APPEND registration_statements
		"        openfpm::metal::register_spirv_kernel(module_words_${index},\n"
		"            sizeof(module_words_${index}) / sizeof(module_words_${index}[0]),\n"
		"            \"${entry}\", ${packed_size}, ${packed_alignment},\n"
		"            module_push_${index},\n"
		"            module_layout_${index}, ${argument_count});\n")
	list(APPEND host_embedded_binaries -Xclang -fcuda-include-gpubinary
		-Xclang "${spirv}")
	math(EXPR index "${index} + 1")
endforeach()

set(output_text
"#include <util/metal/MoltenVKKernel.hpp>\n\n#include <cstddef>\n#include <cstdint>\n\nnamespace\n{\n${module_definitions}")
if (REGISTRATION_FUNCTION)
	string(APPEND output_text
		"}\n\nextern \"C\" void ${REGISTRATION_FUNCTION}()\n{\n"
		"    static bool registered = false;\n"
		"    if (!registered)\n    {\n${registration_statements}"
		"        registered = true;\n    }\n}\n")
else()
	string(APPEND output_text
		"struct module_registrar\n{\n    module_registrar()\n    {\n"
		"${registration_statements}    }\n};\n"
		"const module_registrar register_module;\n}\n")
endif()
file(WRITE "${OUTPUT_CPP}" "${output_text}")

if (HOST_OUTPUT)
	run_checked("HIP host compilation" "${HIP_CLANG}" "@${HOST_FLAGS_FILE}"
		${host_embedded_binaries} "${SOURCE}" -c -o "${HOST_OUTPUT}")
endif()
