// Normalize HIP-generated SPIR kernels to the single physical-pointer argument
// record ABI used by the OpenFPM MoltenVK launcher.
//
// Usage:
//   MoltenVKKernelABI input.ll output.ll metadata.jsonl
//   MoltenVKKernelABI --llvm-compat input.ll output.ll
//   MoltenVKKernelABI --spirv-compat input.spv output.spv
//
// Each output metadata line is an independent JSON object.  Keeping the file
// line-oriented makes it straightforward for CMake to consume without adding
// a JSON library to this small build-time utility.

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

using namespace llvm;

// MoltenVK 1.4.1 embeds a SPIRV-Cross revision with two PhysicalStorageBuffer
// translation bugs which are exposed by valid clspv output.  Keep the fixes in
// this post-clspv compatibility pass: ordinary HIP/CUDA kernels must not carry
// Metal-specific source workarounds.
namespace spirv
{
constexpr std::uint32_t magic = 0x07230203;
constexpr std::uint16_t op_type_int = 21;
constexpr std::uint16_t op_type_float = 22;
constexpr std::uint16_t op_ext_inst_import = 11;
constexpr std::uint16_t op_ext_inst = 12;
constexpr std::uint16_t op_type_pointer = 32;
constexpr std::uint16_t op_constant = 43;
constexpr std::uint16_t op_undef = 45;
constexpr std::uint16_t op_constant_null = 46;
constexpr std::uint16_t op_function_parameter = 55;
constexpr std::uint16_t op_function_call = 57;
constexpr std::uint16_t op_variable = 59;
constexpr std::uint16_t op_load = 61;
constexpr std::uint16_t op_access_chain = 65;
constexpr std::uint16_t op_in_bounds_access_chain = 66;
constexpr std::uint16_t op_ptr_access_chain = 67;
constexpr std::uint16_t op_decorate = 71;
constexpr std::uint16_t op_copy_object = 83;
constexpr std::uint16_t op_convert_u_to_ptr = 120;
constexpr std::uint16_t op_bitcast = 124;
constexpr std::uint16_t op_select = 169;
constexpr std::uint16_t op_atomic_load = 227;
constexpr std::uint16_t op_atomic_store = 228;
constexpr std::uint16_t op_atomic_exchange = 229;
constexpr std::uint16_t op_atomic_compare_exchange = 230;
constexpr std::uint16_t op_atomic_i_increment = 232;
constexpr std::uint16_t op_atomic_xor = 242;
constexpr std::uint16_t op_phi = 245;
constexpr std::uint16_t op_loop_merge = 246;
constexpr std::uint16_t op_selection_merge = 247;
constexpr std::uint16_t op_label = 248;
constexpr std::uint16_t op_branch = 249;
constexpr std::uint16_t op_branch_conditional = 250;
constexpr std::uint16_t op_switch = 251;
constexpr std::uint16_t op_atomic_fadd_ext = 6035;
constexpr std::uint32_t decoration_array_stride = 6;
constexpr std::uint32_t storage_class_physical_buffer = 5349;
}

[[nodiscard]] bool pointerResultInstruction(std::uint16_t opcode)
{
	return opcode == spirv::op_undef ||
		opcode == spirv::op_constant_null ||
		opcode == spirv::op_function_parameter ||
		opcode == spirv::op_function_call ||
		opcode == spirv::op_variable ||
		opcode == spirv::op_load ||
		opcode == spirv::op_access_chain ||
		opcode == spirv::op_in_bounds_access_chain ||
		opcode == spirv::op_ptr_access_chain ||
		opcode == spirv::op_copy_object ||
		opcode == spirv::op_convert_u_to_ptr ||
		opcode == spirv::op_bitcast ||
		opcode == spirv::op_select ||
		opcode == spirv::op_phi;
}

[[nodiscard]] std::uint16_t atomicPointerOperand(std::uint16_t opcode)
{
	if (opcode == spirv::op_atomic_store) return 1;
	if (opcode == spirv::op_atomic_load ||
		opcode == spirv::op_atomic_exchange ||
		opcode == spirv::op_atomic_compare_exchange ||
		(opcode >= spirv::op_atomic_i_increment &&
		 opcode <= spirv::op_atomic_xor) ||
		opcode == spirv::op_atomic_fadd_ext)
		return 3;
	return 0;
}

[[nodiscard]] bool readSpirv(StringRef path,
	std::vector<std::uint32_t> & words)
{
	auto file = MemoryBuffer::getFile(path, false, false);
	if (!file)
	{
		errs() << "MoltenVKKernelABI: cannot read " << path << ": "
			<< file.getError().message() << '\n';
		return false;
	}
	StringRef bytes = file.get()->getBuffer();
	if (bytes.size() < 5 * sizeof(std::uint32_t) ||
		bytes.size() % sizeof(std::uint32_t) != 0)
	{
		errs() << "MoltenVKKernelABI: invalid SPIR-V byte size in " << path
			<< '\n';
		return false;
	}
	words.resize(bytes.size() / sizeof(std::uint32_t));
	std::memcpy(words.data(), bytes.data(), bytes.size());
	if (words.front() != spirv::magic)
	{
		errs() << "MoltenVKKernelABI: invalid SPIR-V magic in " << path << '\n';
		return false;
	}
	return true;
}

[[nodiscard]] bool validInstruction(const std::vector<std::uint32_t> & words,
	std::size_t offset, std::uint16_t & opcode, std::uint16_t & word_count)
{
	if (offset >= words.size()) return false;
	word_count = static_cast<std::uint16_t>(words[offset] >> 16);
	opcode = static_cast<std::uint16_t>(words[offset] & 0xffffu);
	return word_count != 0 && offset + word_count <= words.size();
}

[[nodiscard]] std::string spirvString(
	const std::vector<std::uint32_t> & words, std::size_t first_word,
	std::size_t word_count)
{
	std::string result;
	for (std::size_t i = 0; i < word_count; ++i)
	{
		const std::uint32_t word = words[first_word + i];
		for (unsigned byte = 0; byte < 4; ++byte)
		{
			const char character = static_cast<char>((word >> (byte * 8)) & 0xffu);
			if (character == '\0') return result;
			result.push_back(character);
		}
	}
	return result;
}

[[nodiscard]] int writeSpirvPushConstantMetadata(StringRef input_path,
	StringRef output_path)
{
	std::vector<std::uint32_t> words;
	if (!readSpirv(input_path,words)) return 1;
	std::set<std::uint32_t> reflection_imports;
	std::map<std::uint32_t,std::uint32_t> constants;
	for (std::size_t offset = 5; offset < words.size();)
	{
		std::uint16_t opcode = 0;
		std::uint16_t word_count = 0;
		if (!validInstruction(words,offset,opcode,word_count)) return 1;
		if (opcode == spirv::op_ext_inst_import && word_count >= 3 &&
			spirvString(words,offset + 2,word_count - 2).find(
				"NonSemantic.ClspvReflection") == 0)
			reflection_imports.insert(words[offset + 1]);
		else if (opcode == spirv::op_constant && word_count >= 4)
			constants[words[offset + 2]] = words[offset + 3];
		offset += word_count;
	}

	struct Layout
	{
		std::int64_t argument_pointer = -1;
		std::int64_t global_offset = -1;
		std::int64_t enqueued_local_size = -1;
		std::int64_t global_size = -1;
		std::int64_t region_offset = -1;
		std::int64_t num_workgroups = -1;
		std::int64_t region_group_offset = -1;
		std::uint32_t size = 0;
	} layout;
	auto constant = [&](std::uint32_t id, std::uint32_t & value)
	{
		auto found = constants.find(id);
		if (found == constants.end()) return false;
		value = found->second;
		return true;
	};
	auto record = [&](std::int64_t & destination, std::uint32_t offset,
		std::uint32_t size) -> bool
	{
		if (destination != -1 && destination != offset) return false;
		destination = offset;
		layout.size = std::max(layout.size,offset + size);
		return true;
	};

	for (std::size_t offset = 5; offset < words.size();)
	{
		std::uint16_t opcode = 0;
		std::uint16_t word_count = 0;
		if (!validInstruction(words,offset,opcode,word_count)) return 1;
		if (opcode == spirv::op_ext_inst && word_count >= 7 &&
			reflection_imports.count(words[offset + 3]) != 0)
		{
			const std::uint32_t instruction = words[offset + 4];
			std::uint32_t reflected_offset = 0;
			std::uint32_t reflected_size = 0;
			std::int64_t * destination = nullptr;
			if (instruction >= 15 && instruction <= 20)
			{
				if (!constant(words[offset + 5],reflected_offset) ||
					!constant(words[offset + 6],reflected_size))
					return 1;
				switch (instruction)
				{
				case 15: destination = &layout.global_offset; break;
				case 16: destination = &layout.enqueued_local_size; break;
				case 17: destination = &layout.global_size; break;
				case 18: destination = &layout.region_offset; break;
				case 19: destination = &layout.num_workgroups; break;
				case 20: destination = &layout.region_group_offset; break;
				default: break;
				}
			}
			else if (instruction == 26 && word_count >= 10)
			{
				std::uint32_t ordinal = 0;
				if (!constant(words[offset + 6],ordinal) || ordinal != 0 ||
					!constant(words[offset + 7],reflected_offset) ||
					!constant(words[offset + 8],reflected_size))
					return 1;
				destination = &layout.argument_pointer;
			}
			if (destination != nullptr &&
				!record(*destination,reflected_offset,reflected_size))
				return 1;
		}
		offset += word_count;
	}
	layout.size = static_cast<std::uint32_t>(alignTo(layout.size,4u));

	std::error_code error;
	raw_fd_ostream output(output_path,error,sys::fs::OF_Text);
	if (error)
	{
		errs() << "MoltenVKKernelABI: cannot write " << output_path << ": "
			<< error.message() << '\n';
		return 1;
	}
	output << "{\"argument_pointer\":" << layout.argument_pointer
		<< ",\"global_offset\":" << layout.global_offset
		<< ",\"enqueued_local_size\":" << layout.enqueued_local_size
		<< ",\"global_size\":" << layout.global_size
		<< ",\"region_offset\":" << layout.region_offset
		<< ",\"num_workgroups\":" << layout.num_workgroups
		<< ",\"region_group_offset\":" << layout.region_group_offset
		<< ",\"size\":" << layout.size << "}\n";
	return 0;
}

[[nodiscard]] int normalizeMoltenVKSpirv(StringRef input_path,
	StringRef output_path)
{
	std::vector<std::uint32_t> words;
	if (!readSpirv(input_path, words)) return 1;

	std::map<std::uint32_t, std::uint32_t> pointer_pointee_types;
	std::map<std::uint32_t, std::uint32_t> pointer_storage_classes;
	std::map<std::uint32_t, std::uint32_t> type_sizes;
	std::map<std::uint32_t, std::uint32_t> value_types;
	std::map<std::uint32_t, std::uint32_t> all_value_types;
	std::set<std::uint32_t> integer_types;
	std::set<std::uint32_t> physical_pointer_types;
	std::set<std::uint32_t> array_stride_types;
	std::uint32_t zero_integer = 0;
	std::size_t first_type = words.size();

	struct AtomicRepair
	{
		std::uint32_t pointer_type = 0;
		std::uint32_t pointer = 0;
		std::uint16_t pointer_operand = 0;
		std::uint32_t value_type = 0;
		bool use_access_chain = false;
		bool use_bitcast = false;
		bool use_direct_pointer = false;
	};
	std::map<std::size_t, AtomicRepair> atomic_repairs;

	for (std::size_t offset = 5; offset < words.size();)
	{
		std::uint16_t opcode = 0;
		std::uint16_t word_count = 0;
		if (!validInstruction(words, offset, opcode, word_count))
		{
			errs() << "MoltenVKKernelABI: malformed SPIR-V instruction at word "
				<< offset << '\n';
			return 1;
		}
		if (opcode >= 19 && opcode <= 39 && first_type == words.size())
			first_type = offset;
		if ((opcode == spirv::op_type_int || opcode == spirv::op_type_float) &&
			word_count >= 3 && words[offset + 2] % 8 == 0)
		{
			type_sizes[words[offset + 1]] = words[offset + 2] / 8;
			if (opcode == spirv::op_type_int)
				integer_types.insert(words[offset + 1]);
		}
		else if (opcode == spirv::op_type_pointer && word_count == 4)
		{
			pointer_pointee_types[words[offset + 1]] = words[offset + 3];
			pointer_storage_classes[words[offset + 1]] = words[offset + 2];
			if (words[offset + 2] == spirv::storage_class_physical_buffer)
				physical_pointer_types.insert(words[offset + 1]);
		}
		else if (opcode == spirv::op_decorate && word_count >= 4 &&
			words[offset + 2] == spirv::decoration_array_stride)
			array_stride_types.insert(words[offset + 1]);
		else if (opcode == spirv::op_constant && word_count >= 4 &&
			integer_types.count(words[offset + 1]) != 0)
		{
			bool is_zero = true;
			for (std::uint16_t i = 3; i < word_count; ++i)
				is_zero = is_zero && words[offset + i] == 0;
			if (is_zero && (zero_integer == 0 ||
				type_sizes[words[offset + 1]] == 4))
				zero_integer = words[offset + 2];
		}
		if (word_count >= 3 && pointerResultInstruction(opcode) &&
			pointer_pointee_types.count(words[offset + 1]) != 0)
		{
			all_value_types[words[offset + 2]] = words[offset + 1];
			if (physical_pointer_types.count(words[offset + 1]) != 0)
				value_types[words[offset + 2]] = words[offset + 1];
		}

		const std::uint16_t pointer_operand = atomicPointerOperand(opcode);
		if (pointer_operand != 0)
		{
			if (word_count <= pointer_operand)
			{
				errs() << "MoltenVKKernelABI: malformed atomic instruction\n";
				return 1;
			}
			const std::uint32_t pointer = words[offset + pointer_operand];
			auto pointer_type = value_types.find(pointer);
			if (pointer_type != value_types.end())
				atomic_repairs[offset] =
					{pointer_type->second, pointer, pointer_operand,
						pointer_operand == 1 ? 0 : words[offset + 1], false, false};
			else if (auto any_pointer_type = all_value_types.find(pointer);
				any_pointer_type != all_value_types.end())
				atomic_repairs[offset] =
					{any_pointer_type->second, pointer, pointer_operand,
						pointer_operand == 1 ? 0 : words[offset + 1], false, false};
		}
		offset += word_count;
	}

	struct PointerAccessRepair
	{
		std::uint32_t original_type = 0;
		std::uint32_t original_result = 0;
		std::uint32_t corrected_type = 0;
		std::uint32_t corrected_result = 0;
		bool emit_original_bitcast = true;
		bool use_access_chain = false;
	};
	struct PhiPointerCast
	{
		std::uint32_t target_type = 0;
		std::uint32_t result = 0;
		std::uint32_t source = 0;
	};
	struct PointerOperandCast
	{
		std::uint32_t target_type = 0;
		std::uint32_t result = 0;
		std::uint32_t source = 0;
		std::uint16_t operand = 0;
	};
	std::map<std::size_t, PointerAccessRepair> pointer_access_repairs;
	std::map<std::size_t, PointerOperandCast> pointer_operand_casts;
	std::map<std::size_t, std::map<std::uint16_t, std::uint32_t>>
		phi_operand_repairs;
	std::map<std::uint32_t, std::vector<PhiPointerCast>> phi_casts_by_block;
	std::uint32_t next_id = words[3];
	std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>
		synthesized_pointer_types;
	auto pointerTypeFor = [&](std::uint32_t storage,
		std::uint32_t pointee) -> std::uint32_t
	{
		for (const auto & [candidate, candidate_pointee] : pointer_pointee_types)
			if (candidate_pointee == pointee &&
				pointer_storage_classes[candidate] == storage)
				return candidate;
		const auto key = std::make_pair(storage, pointee);
		auto existing = synthesized_pointer_types.find(key);
		if (existing != synthesized_pointer_types.end()) return existing->second;
		const std::uint32_t type = next_id++;
		synthesized_pointer_types[key] = type;
		pointer_pointee_types[type] = pointee;
		pointer_storage_classes[type] = storage;
		if (storage == spirv::storage_class_physical_buffer)
			physical_pointer_types.insert(type);
		return type;
	};
	for (std::size_t offset = 5; offset < words.size();)
	{
		std::uint16_t opcode = 0;
		std::uint16_t word_count = 0;
		if (!validInstruction(words, offset, opcode, word_count)) return 1;
		if (opcode == spirv::op_ptr_access_chain && word_count == 5)
		{
			const std::uint32_t result_type = words[offset + 1];
			const std::uint32_t base = words[offset + 3];
			auto base_type = all_value_types.find(base);
			if (base_type != all_value_types.end() &&
				base_type->second != result_type &&
				pointer_storage_classes[base_type->second] ==
					pointer_storage_classes[result_type] &&
				pointer_pointee_types[base_type->second] !=
					pointer_pointee_types[result_type])
			{
				pointer_access_repairs[offset] = {result_type,
					words[offset + 2], base_type->second, next_id++, true, false};
			}
		}
		else if (opcode == spirv::op_load && word_count >= 4)
		{
			const std::uint32_t result_type = words[offset + 1];
			const std::uint32_t pointer = words[offset + 3];
			auto pointer_type = all_value_types.find(pointer);
			if (pointer_type != all_value_types.end() &&
				pointer_storage_classes[pointer_type->second] ==
					spirv::storage_class_physical_buffer &&
				pointer_pointee_types[pointer_type->second] != result_type)
			{
				const std::uint32_t target_type = pointerTypeFor(
					spirv::storage_class_physical_buffer, result_type);
				pointer_operand_casts[offset] =
					{target_type, next_id++, pointer, 3};
			}
		}
		else if (opcode == spirv::op_phi && word_count >= 5)
		{
			const std::uint32_t result_type = words[offset + 1];
			if (pointer_pointee_types.count(result_type) != 0)
			{
				for (std::uint16_t operand = 3; operand + 1 < word_count;
					operand += 2)
				{
					const std::uint32_t incoming = words[offset + operand];
					auto incoming_type = all_value_types.find(incoming);
					if (incoming_type == all_value_types.end() ||
						incoming_type->second == result_type ||
						pointer_storage_classes[incoming_type->second] !=
							pointer_storage_classes[result_type])
						continue;
					const std::uint32_t cast_result = next_id++;
					phi_operand_repairs[offset][operand] = cast_result;
					phi_casts_by_block[words[offset + operand + 1]].push_back(
						{result_type, cast_result, incoming});
				}
			}
		}
		offset += word_count;
	}
	std::map<std::uint32_t, PointerAccessRepair *> access_repair_by_result;
	for (auto & [offset, repair] : pointer_access_repairs)
	{
		(void)offset;
		access_repair_by_result[repair.original_result] = &repair;
	}
	for (auto & [offset, repair] : atomic_repairs)
	{
		(void)offset;
		auto access = access_repair_by_result.find(repair.pointer);
		if (access == access_repair_by_result.end() &&
			physical_pointer_types.count(repair.pointer_type) == 0)
		{
			// clspv already emitted the canonical pointer for ordinary logical
			// atomics (notably scalar Workgroup variables).  Adding a zero
			// PtrAccessChain here makes older SPIRV-Cross generate scalar[0],
			// which is invalid MSL.  Only physical pointers and repaired
			// mis-typed access chains need an adapter instruction.
			repair.use_direct_pointer = true;
			continue;
		}
		if (repair.value_type == 0) continue;
		const std::uint32_t source_pointer_type =
			access != access_repair_by_result.end() ?
				access->second->corrected_type : repair.pointer_type;
		const std::uint32_t storage =
			pointer_storage_classes[source_pointer_type];
		const std::uint32_t scalar_pointer_type =
			pointerTypeFor(storage, repair.value_type);
		if (scalar_pointer_type != 0)
		{
			repair.pointer_type = scalar_pointer_type;
			if (access != access_repair_by_result.end())
			{
				repair.pointer = access->second->corrected_result;
				// For logical Workgroup arrays, clspv has encoded shared[index] as
				// a mis-typed PtrAccessChain.  Pointer arithmetic on the array type
				// means (&shared)[index], which older SPIRV-Cross faithfully turns
				// into an out-of-bounds array-of-arrays access.  Restore the source
				// operation directly as OpAccessChain shared,index. Physical buffer
				// repairs still require the two-stage pointer adapter below.
				if (storage != spirv::storage_class_physical_buffer)
				{
					access->second->corrected_type = scalar_pointer_type;
					access->second->emit_original_bitcast = false;
					access->second->use_access_chain = true;
					repair.use_direct_pointer = true;
				}
				else
					repair.use_access_chain = true;
			}
			else if (pointer_pointee_types[source_pointer_type] != repair.value_type &&
				storage == spirv::storage_class_physical_buffer)
				repair.use_bitcast = true;
		}
	}

	const bool needs_atomic_pointer_adapter = std::any_of(
		atomic_repairs.begin(),atomic_repairs.end(),
		[](const auto & item) { return !item.second.use_direct_pointer; });
	if (needs_atomic_pointer_adapter && zero_integer == 0)
	{
		errs() << "MoltenVKKernelABI: cannot materialize an atomic pointer "
			"without an integer zero constant\n";
		return 1;
	}
	std::map<std::uint32_t, std::uint32_t> missing_array_strides;
	for (const auto & [offset, repair] : atomic_repairs)
	{
		(void)offset;
		if (repair.use_access_chain || repair.use_bitcast ||
			physical_pointer_types.count(repair.pointer_type) == 0)
			continue;
		if (array_stride_types.count(repair.pointer_type) != 0) continue;
		auto pointee = pointer_pointee_types.find(repair.pointer_type);
		if (pointee == pointer_pointee_types.end() ||
			type_sizes.count(pointee->second) == 0)
		{
			errs() << "MoltenVKKernelABI: cannot determine atomic pointer stride\n";
			return 1;
		}
		missing_array_strides[repair.pointer_type] = type_sizes[pointee->second];
	}

	std::vector<std::uint32_t> output;
	output.reserve(words.size() + missing_array_strides.size() * 4 +
		atomic_repairs.size() * 5 + pointer_access_repairs.size() * 4 +
		phi_operand_repairs.size() * 4 + pointer_operand_casts.size() * 4 +
		synthesized_pointer_types.size() * 4);
	output.insert(output.end(), words.begin(), words.begin() + 5);
	std::uint32_t current_block = 0;
	std::set<std::uint32_t> emitted_phi_cast_blocks;
	for (std::size_t offset = 5; offset < words.size();)
	{
		std::uint16_t opcode = 0;
		std::uint16_t word_count = 0;
		if (!validInstruction(words, offset, opcode, word_count)) return 1;
		if (offset == first_type)
		{
			for (const auto & [pointer_type, stride] : missing_array_strides)
			{
				output.push_back((4u << 16) | spirv::op_decorate);
				output.push_back(pointer_type);
				output.push_back(spirv::decoration_array_stride);
				output.push_back(stride);
			}
		}
		if (opcode == spirv::op_label && word_count >= 2)
			current_block = words[offset + 1];
		const bool before_block_exit =
			opcode == spirv::op_loop_merge ||
			opcode == spirv::op_selection_merge ||
			(opcode >= spirv::op_branch && opcode <= 255);
		if (before_block_exit && current_block != 0 &&
			emitted_phi_cast_blocks.insert(current_block).second)
		{
			auto casts = phi_casts_by_block.find(current_block);
			if (casts != phi_casts_by_block.end())
			{
				for (const PhiPointerCast & cast : casts->second)
				{
					output.push_back((4u << 16) | spirv::op_bitcast);
					output.push_back(cast.target_type);
					output.push_back(cast.result);
					output.push_back(cast.source);
				}
			}
		}

		const auto atomic = atomic_repairs.find(offset);
		std::uint32_t atomic_pointer = 0;
		if (atomic != atomic_repairs.end())
		{
			if (!atomic->second.use_direct_pointer)
			{
				atomic_pointer = next_id++;
				if (atomic->second.use_bitcast)
				{
					output.push_back((4u << 16) | spirv::op_bitcast);
					output.push_back(atomic->second.pointer_type);
					output.push_back(atomic_pointer);
					output.push_back(atomic->second.pointer);
				}
				else
				{
					output.push_back((5u << 16) |
						(atomic->second.use_access_chain ? spirv::op_access_chain :
							spirv::op_ptr_access_chain));
					output.push_back(atomic->second.pointer_type);
					output.push_back(atomic_pointer);
					output.push_back(atomic->second.pointer);
					output.push_back(zero_integer);
				}
			}
			else
			{
				// Usually this is the original already-canonical pointer.  A
				// logical-array repair instead points at the scalar access chain
				// emitted when its malformed clspv instruction is rewritten.
				atomic_pointer = atomic->second.pointer;
			}
		}

		const auto pointer_operand_cast = pointer_operand_casts.find(offset);
		if (pointer_operand_cast != pointer_operand_casts.end())
		{
			output.push_back((4u << 16) | spirv::op_bitcast);
			output.push_back(pointer_operand_cast->second.target_type);
			output.push_back(pointer_operand_cast->second.result);
			output.push_back(pointer_operand_cast->second.source);
		}
		const std::size_t instruction_start = output.size();
		const auto pointer_access = pointer_access_repairs.find(offset);
		if (pointer_access != pointer_access_repairs.end())
		{
			output.insert(output.end(), words.begin() + offset,
				words.begin() + offset + word_count);
			if (pointer_access->second.use_access_chain)
				output[instruction_start] =
					(static_cast<std::uint32_t>(word_count) << 16) |
					spirv::op_access_chain;
			output[instruction_start + 1] = pointer_access->second.corrected_type;
			output[instruction_start + 2] = pointer_access->second.corrected_result;
			if (pointer_access->second.emit_original_bitcast)
			{
				output.push_back((4u << 16) | spirv::op_bitcast);
				output.push_back(pointer_access->second.original_type);
				output.push_back(pointer_access->second.original_result);
				output.push_back(pointer_access->second.corrected_result);
			}
		}
		else
		{
			output.insert(output.end(), words.begin() + offset,
				words.begin() + offset + word_count);
		}
		if ((opcode == spirv::op_access_chain ||
			opcode == spirv::op_in_bounds_access_chain) && word_count == 4)
			output[instruction_start] =
				(4u << 16) | spirv::op_copy_object;
		const auto phi_repair = phi_operand_repairs.find(offset);
		if (phi_repair != phi_operand_repairs.end())
			for (const auto & [operand, replacement] : phi_repair->second)
				output[instruction_start + operand] = replacement;
		if (pointer_operand_cast != pointer_operand_casts.end())
			output[instruction_start + pointer_operand_cast->second.operand] =
				pointer_operand_cast->second.result;
		if (atomic_pointer != 0)
			output[instruction_start + atomic->second.pointer_operand] =
				atomic_pointer;
		if (word_count >= 2 && opcode >= 19 && opcode <= 39)
			for (const auto & [key, type] : synthesized_pointer_types)
				if (words[offset + 1] == key.second)
				{
					output.push_back((4u << 16) | spirv::op_type_pointer);
					output.push_back(type);
					output.push_back(key.first);
					output.push_back(key.second);
				}
		offset += word_count;
	}
	output[3] = next_id;

	std::error_code error;
	raw_fd_ostream stream(output_path, error, sys::fs::OF_None);
	if (error)
	{
		errs() << "MoltenVKKernelABI: cannot write " << output_path << ": "
			<< error.message() << '\n';
		return 1;
	}
	stream.write(reinterpret_cast<const char *>(output.data()),
		output.size() * sizeof(std::uint32_t));
	stream.close();
	return 0;
}

struct ArgumentLayout
{
	std::uint64_t offset = 0;
	std::uint64_t size = 0;
	std::uint64_t alignment = 1;
};

struct KernelLayout
{
	std::string entry;
	std::uint64_t size = 0;
	std::uint64_t alignment = 1;
	std::vector<ArgumentLayout> arguments;
};

[[nodiscard]] bool fixedSize(TypeSize size, std::uint64_t & result,
	StringRef description)
{
	if (size.isScalable())
	{
		errs() << "MoltenVK ABI cannot represent scalable " << description << '\n';
		return false;
	}
	result = size.getFixedValue();
	return true;
}

void writeJsonString(raw_ostream & output, StringRef value)
{
	static constexpr char hexadecimal[] = "0123456789abcdef";
	output << '"';
	for (unsigned char character : value.bytes())
	{
		switch (character)
		{
		case '"': output << "\\\""; break;
		case '\\': output << "\\\\"; break;
		case '\b': output << "\\b"; break;
		case '\f': output << "\\f"; break;
		case '\n': output << "\\n"; break;
		case '\r': output << "\\r"; break;
		case '\t': output << "\\t"; break;
		default:
			if (character < 0x20)
			{
				output << "\\u00" << hexadecimal[character >> 4]
					<< hexadecimal[character & 0x0f];
			}
			else
			{
				output << static_cast<char>(character);
			}
		}
	}
	output << '"';
}

void writeMetadata(raw_ostream & output, const KernelLayout & layout)
{
	output << "{\"entry\":";
	writeJsonString(output, layout.entry);
	output << ",\"size\":" << layout.size
		<< ",\"alignment\":" << layout.alignment
		<< ",\"arguments\":[";
	for (std::size_t i = 0; i < layout.arguments.size(); ++i)
	{
		if (i != 0) output << ',';
		const auto & argument = layout.arguments[i];
		output << "{\"index\":" << i
			<< ",\"offset\":" << argument.offset
			<< ",\"size\":" << argument.size
			<< ",\"alignment\":" << argument.alignment << '}';
	}
	output << "]}\n";
}

[[nodiscard]] Type * packedFieldType(Argument & argument,
	LLVMContext & context)
{
	if (argument.hasByValAttr())
		return argument.getParamByValType();
	if (argument.getType()->isPointerTy())
		return Type::getInt64Ty(context);
	return argument.getType();
}

// clspv's physical-storage-buffer ABI is not the C++ ABI. In particular, a
// whole aggregate load can be scalarized as though nested LLVM structs were
// packed, dropping padding which clang included in the host argument object.
// Reconstruct byval values leaf-by-leaf at explicit DataLayout byte offsets so
// the same HIP/CUDA object representation is used on both sides of the launch.
[[nodiscard]] Value * loadPackedValue(Type * type, Value * packed_argument,
	std::uint64_t byte_offset, Align packed_alignment,
	const DataLayout & data_layout, IRBuilder<> & builder)
{
	if (auto * structure = dyn_cast<StructType>(type))
	{
		if (structure->isOpaque()) return nullptr;
		Value * value = UndefValue::get(type);
		const StructLayout * layout = data_layout.getStructLayout(structure);
		for (unsigned i = 0; i < structure->getNumElements(); ++i)
		{
			Value * field = loadPackedValue(structure->getElementType(i),
				packed_argument, byte_offset + layout->getElementOffset(i),
				packed_alignment, data_layout, builder);
			if (field == nullptr) return nullptr;
			value = builder.CreateInsertValue(value, field, {i},
				"openfpm_packed_field");
		}
		return value;
	}
	if (auto * array = dyn_cast<ArrayType>(type))
	{
		std::uint64_t stride = 0;
		if (!fixedSize(data_layout.getTypeAllocSize(array->getElementType()),
			stride, "packed array element")) return nullptr;
		Value * value = UndefValue::get(type);
		for (std::uint64_t i = 0; i < array->getNumElements(); ++i)
		{
			Value * element = loadPackedValue(array->getElementType(),
				packed_argument, byte_offset + i * stride, packed_alignment,
				data_layout, builder);
			if (element == nullptr) return nullptr;
			value = builder.CreateInsertValue(value, element,
				{static_cast<unsigned>(i)}, "openfpm_packed_element");
		}
		return value;
	}
	if (auto * vector = dyn_cast<VectorType>(type))
	{
		const ElementCount count = vector->getElementCount();
		if (count.isScalable()) return nullptr;
		std::uint64_t stride = 0;
		if (!fixedSize(data_layout.getTypeStoreSize(vector->getElementType()),
			stride, "packed vector element")) return nullptr;
		Value * value = UndefValue::get(type);
		for (unsigned i = 0; i < count.getFixedValue(); ++i)
		{
			Value * element = loadPackedValue(vector->getElementType(),
				packed_argument, byte_offset + i * stride, packed_alignment,
				data_layout, builder);
			if (element == nullptr) return nullptr;
			value = builder.CreateInsertElement(value, element, builder.getInt32(i),
				"openfpm_packed_vector_element");
		}
		return value;
	}
	if (!type->isFirstClassType() || !type->isSized()) return nullptr;

	Value * address = builder.CreateInBoundsGEP(Type::getInt8Ty(builder.getContext()),
		packed_argument, builder.getInt64(byte_offset), "openfpm_packed_address");
	const Align alignment = commonAlignment(packed_alignment, byte_offset);
	if (auto * pointer = dyn_cast<PointerType>(type))
	{
		IntegerType * integer_type = IntegerType::get(builder.getContext(),
			data_layout.getPointerSizeInBits(pointer->getAddressSpace()));
		Value * integer = builder.CreateAlignedLoad(integer_type, address,
			alignment, "openfpm_packed_pointer_bits");
		return builder.CreateIntToPtr(integer, pointer,
			"openfpm_packed_pointer");
	}
	if (type->isFloatingPointTy())
	{
		std::uint64_t byte_size = 0;
		if (!fixedSize(data_layout.getTypeStoreSize(type),byte_size,
			"packed floating-point scalar")) return nullptr;
		IntegerType * bits_type = IntegerType::get(builder.getContext(),
			static_cast<unsigned>(byte_size * 8));
		Value * bits = builder.CreateAlignedLoad(bits_type,address,alignment,
			"openfpm_packed_float_bits");
		return builder.CreateBitCast(bits,type,"openfpm_packed_float");
	}
	return builder.CreateAlignedLoad(type, address, alignment,
		"openfpm_packed_scalar");
}

[[nodiscard]] bool isChipStarIntegerAtomicAdd(StringRef name)
{
	return name == "__chip_atomic_add_i" ||
		name == "__chip_atomic_add_u" ||
		name == "__chip_atomic_add_l" ||
		name == "__chip_atomic_add_system_i" ||
		name == "__chip_atomic_add_system_u" ||
		name == "__chip_atomic_add_system_l" ||
		name == "__chip_atomic_add_block_i" ||
		name == "__chip_atomic_add_block_u" ||
		name == "__chip_atomic_add_block_l";
}

[[nodiscard]] bool isChipStarFloatAtomicAdd(StringRef name)
{
	return name == "__chip_atomic_add_f32" ||
		name == "__chip_atomic_add_system_f32";
}

[[nodiscard]] bool containsPointerLeaf(Type * type,
	SmallPtrSetImpl<Type *> & visiting)
{
	if (type->isPointerTy()) return true;
	if (!visiting.insert(type).second) return false;
	bool result = false;
	if (auto * structure = dyn_cast<StructType>(type))
	{
		if (!structure->isOpaque())
			for (Type * element : structure->elements())
				if (containsPointerLeaf(element, visiting))
				{
					result = true;
					break;
				}
	}
	else if (auto * array = dyn_cast<ArrayType>(type))
		result = containsPointerLeaf(array->getElementType(), visiting);
	else if (auto * vector = dyn_cast<VectorType>(type))
		result = containsPointerLeaf(vector->getElementType(), visiting);
	visiting.erase(type);
	return result;
}

[[nodiscard]] bool containsPointerLeaf(Type * type)
{
	SmallPtrSet<Type *, 8> visiting;
	return containsPointerLeaf(type, visiting);
}

// A common HIP loop keeps a PhysicalStorageBuffer pointer in an AS4 phi:
// the entry value is an AS1 -> AS4 cast and the back edge is a byte GEP rooted
// in that same phi.  Once inlined, the phi itself no longer exposes a simple
// cast chain to recoverTypedPhysicalPointer.  clspv then lowers a 4-byte load
// through the generic pointer using the pointee's stronger 8-byte alignment,
// which can discard the low address bit for odd indices.  Prove this narrow
// loop shape before treating the phi's runtime value as the physical address.
[[nodiscard]] bool isDerivedFromPhi(Value * value, PHINode * phi,
	unsigned depth = 0)
{
	if (value == phi) return true;
	if (depth > 32 || value == nullptr || !value->getType()->isPointerTy())
		return false;
	if (auto * gep = dyn_cast<GetElementPtrInst>(value))
		return isDerivedFromPhi(gep->getPointerOperand(),phi,depth+1);
	if (auto * bitcast = dyn_cast<BitCastInst>(value))
		return isDerivedFromPhi(bitcast->getOperand(0),phi,depth+1);
	if (auto * freeze = dyn_cast<FreezeInst>(value))
		return isDerivedFromPhi(freeze->getOperand(0),phi,depth+1);
	if (auto * selection = dyn_cast<SelectInst>(value))
		return isDerivedFromPhi(selection->getTrueValue(),phi,depth+1) &&
			isDerivedFromPhi(selection->getFalseValue(),phi,depth+1);
	return false;
}

[[nodiscard]] bool isPhysicalLoopPhi(PHINode * phi);

[[nodiscard]] bool isPhysicalGenericSeed(Value * value, unsigned depth = 0)
{
	if (depth > 32 || value == nullptr || !value->getType()->isPointerTy())
		return false;
	if (auto * cast = dyn_cast<AddrSpaceCastInst>(value))
	{
		Value * source = cast->getOperand(0);
		return source->getType()->isPointerTy() &&
			source->getType()->getPointerAddressSpace() == 1 &&
			cast->getType()->getPointerAddressSpace() == 4;
	}
	if (auto * conversion = dyn_cast<IntToPtrInst>(value))
		return conversion->getType()->getPointerAddressSpace() == 4;
	if (auto * bitcast = dyn_cast<BitCastInst>(value))
		return isPhysicalGenericSeed(bitcast->getOperand(0),depth+1);
	if (auto * freeze = dyn_cast<FreezeInst>(value))
		return isPhysicalGenericSeed(freeze->getOperand(0),depth+1);
	if (auto * gep = dyn_cast<GetElementPtrInst>(value))
		return isPhysicalGenericSeed(gep->getPointerOperand(),depth+1);
	if (auto * selection = dyn_cast<SelectInst>(value))
		return isPhysicalGenericSeed(selection->getTrueValue(),depth+1) &&
			isPhysicalGenericSeed(selection->getFalseValue(),depth+1);
	if (auto * phi = dyn_cast<PHINode>(value))
		return isPhysicalLoopPhi(phi);
	if (isa<ConstantPointerNull>(value))
		return value->getType()->getPointerAddressSpace() == 4;
	return false;
}

[[nodiscard]] bool isPhysicalLoopPhi(PHINode * phi)
{
	if (phi == nullptr || phi->getType()->getPointerAddressSpace() != 4)
		return false;
	bool has_physical_seed = false;
	for (Value * incoming : phi->incoming_values())
	{
		if (isDerivedFromPhi(incoming,phi)) continue;
		if (isPhysicalGenericSeed(incoming))
		{
			has_physical_seed = true;
			continue;
		}
		return false;
	}
	return has_physical_seed;
}

// SROA commonly leaves a pointer-valued extractvalue rooted in the aggregate
// assembled by loadPackedValue. Recover the inserted scalar so the physical
// pointer pass sees the original inttoptr instead of an opaque AS4 value.
[[nodiscard]] Value * findInsertedAggregateValue(Value * aggregate,
	ArrayRef<unsigned> indices, SmallVectorImpl<Instruction *> & dead_chain,
	unsigned depth = 0)
{
	if (aggregate == nullptr || indices.empty() || depth > 64) return nullptr;
	auto * insertion = dyn_cast<InsertValueInst>(aggregate);
	if (insertion == nullptr) return nullptr;
	dead_chain.push_back(insertion);

	const ArrayRef<unsigned> inserted_indices = insertion->getIndices();
	const bool inserted_is_prefix = inserted_indices.size() <= indices.size() &&
		std::equal(inserted_indices.begin(),inserted_indices.end(),indices.begin());
	if (inserted_is_prefix)
	{
		Value * inserted = insertion->getInsertedValueOperand();
		if (inserted_indices.size() == indices.size()) return inserted;
		return findInsertedAggregateValue(inserted,
			indices.drop_front(inserted_indices.size()),dead_chain,depth + 1);
	}
	return findInsertedAggregateValue(insertion->getAggregateOperand(),indices,
		dead_chain,depth + 1);
}

// Collect the byte offset represented by a pointer chain rooted at an
// AS1 -> AS4 cast.  Continue through AS1 GEPs below the cast as clang commonly
// places an array subscript immediately before the generic conversion.
[[nodiscard]] bool collectPhysicalPointerPath(Value * value,
	const DataLayout & data_layout, unsigned index_width,
	SmallMapVector<Value *, APInt, 4> & variable_offsets,
	APInt & constant_offset, SmallVectorImpl<Instruction *> & dead_chain,
	Value *& root, bool & crossed_generic_cast, unsigned depth = 0)
{
	if (depth > 32 || !value->getType()->isPointerTy()) return false;
	if (auto * gep = dyn_cast<GetElementPtrInst>(value))
	{
		if (data_layout.getIndexSizeInBits(
			gep->getPointerAddressSpace()) != index_width)
			return false;
		dead_chain.push_back(gep);
		if (!collectPhysicalPointerPath(gep->getPointerOperand(), data_layout,
			index_width, variable_offsets, constant_offset, dead_chain, root,
			crossed_generic_cast, depth + 1))
			return false;

		SmallMapVector<Value *, APInt, 4> gep_variables;
		APInt gep_constant(index_width, 0);
		if (!gep->collectOffset(data_layout, index_width, gep_variables,
			gep_constant))
			return false;
		constant_offset += gep_constant;
		for (auto & [index, multiplier] : gep_variables)
		{
			auto iterator = variable_offsets.insert(
				{index, APInt(index_width, 0)}).first;
			iterator->second += multiplier;
		}
		return true;
	}
	if (auto * extract = dyn_cast<ExtractValueInst>(value))
	{
		dead_chain.push_back(extract);
		Value * inserted = findInsertedAggregateValue(
			extract->getAggregateOperand(),extract->getIndices(),dead_chain);
		if (inserted == nullptr || !inserted->getType()->isPointerTy()) return false;
		return collectPhysicalPointerPath(inserted,data_layout,index_width,
			variable_offsets,constant_offset,dead_chain,root,
			crossed_generic_cast,depth + 1);
	}
	if (auto * cast = dyn_cast<AddrSpaceCastInst>(value))
	{
		Value * source = cast->getOperand(0);
		if (!source->getType()->isPointerTy() ||
			source->getType()->getPointerAddressSpace() != 1 ||
			cast->getType()->getPointerAddressSpace() != 4)
			return false;
		dead_chain.push_back(cast);
		crossed_generic_cast = true;
		return collectPhysicalPointerPath(source, data_layout, index_width,
			variable_offsets, constant_offset, dead_chain, root,
			crossed_generic_cast, depth + 1);
	}
	if (auto * cast = dyn_cast<BitCastInst>(value))
	{
		dead_chain.push_back(cast);
		return collectPhysicalPointerPath(cast->getOperand(0), data_layout,
			index_width, variable_offsets, constant_offset, dead_chain, root,
			crossed_generic_cast, depth + 1);
	}
	if (auto * freeze = dyn_cast<FreezeInst>(value))
	{
		dead_chain.push_back(freeze);
		return collectPhysicalPointerPath(freeze->getOperand(0), data_layout,
			index_width, variable_offsets, constant_offset, dead_chain, root,
			crossed_generic_cast, depth + 1);
	}
	if (auto * phi = dyn_cast<PHINode>(value); isPhysicalLoopPhi(phi))
	{
		root = phi;
		return true;
	}
	if (auto * selection = dyn_cast<SelectInst>(value);
		selection != nullptr &&
		selection->getType()->getPointerAddressSpace() == 4 &&
		isPhysicalGenericSeed(selection->getTrueValue()) &&
		isPhysicalGenericSeed(selection->getFalseValue()))
	{
		root = selection;
		return true;
	}
	if (isa<ConstantPointerNull>(value) &&
		value->getType()->getPointerAddressSpace() == 4)
	{
		root = value;
		return true;
	}
	if (isa<IntToPtrInst>(value) &&
		(value->getType()->getPointerAddressSpace() == 1 ||
		 value->getType()->getPointerAddressSpace() == 4))
	{
		root = value;
		return true;
	}
	if (crossed_generic_cast &&
		value->getType()->getPointerAddressSpace() == 1)
	{
		root = value;
		return true;
	}
	return false;
}

// Rebuild a proven physical AS4 control-flow value as an AS1 pointer.  This is
// needed for loop-carried pointer phis: asking clspv to infer an address space
// through the generic phi can either mix AS1/AS4 select arms or recurse in its
// LowerAddrSpaceCast pass.  The numeric address and byte GEPs are unchanged.
[[nodiscard]] Value * rebuildPhysicalPointerControlFlow(Value * value,
	Module & module, IRBuilder<> & builder,
	std::map<PHINode *, PHINode *> & rebuilt_phis,
	SmallVectorImpl<Instruction *> & dead_chain, unsigned depth = 0)
{
	if (value == nullptr || !value->getType()->isPointerTy() || depth > 64)
		return nullptr;
	if (value->getType()->getPointerAddressSpace() == 1) return value;
	if (value->getType()->getPointerAddressSpace() != 4) return nullptr;

	PointerType * physical_type = PointerType::get(module.getContext(),1);
	if (auto * conversion = dyn_cast<IntToPtrInst>(value))
	{
		dead_chain.push_back(conversion);
		return builder.CreateIntToPtr(conversion->getOperand(0),physical_type,
			"openfpm_physical_pointer");
	}
	if (auto * cast = dyn_cast<AddrSpaceCastInst>(value))
	{
		dead_chain.push_back(cast);
		return rebuildPhysicalPointerControlFlow(cast->getOperand(0),module,
			builder,rebuilt_phis,dead_chain,depth+1);
	}
	if (auto * cast = dyn_cast<BitCastInst>(value))
	{
		dead_chain.push_back(cast);
		return rebuildPhysicalPointerControlFlow(cast->getOperand(0),module,
			builder,rebuilt_phis,dead_chain,depth+1);
	}
	if (auto * freeze = dyn_cast<FreezeInst>(value))
	{
		dead_chain.push_back(freeze);
		Value * operand = rebuildPhysicalPointerControlFlow(freeze->getOperand(0),
			module,builder,rebuilt_phis,dead_chain,depth+1);
		if (operand == nullptr) return nullptr;
		IntegerType * address_type = IntegerType::get(module.getContext(),
			module.getDataLayout().getIndexSizeInBits(1));
		Value * address = builder.CreatePtrToInt(operand,address_type,
			"openfpm_physical_address");
		address = builder.CreateFreeze(address,"openfpm_physical_address_frozen");
		return builder.CreateIntToPtr(address,physical_type,
			"openfpm_physical_pointer");
	}
	if (auto * gep = dyn_cast<GetElementPtrInst>(value))
	{
		dead_chain.push_back(gep);
		Value * base = rebuildPhysicalPointerControlFlow(gep->getPointerOperand(),
			module,builder,rebuilt_phis,dead_chain,depth+1);
		if (base == nullptr) return nullptr;

		const DataLayout & data_layout = module.getDataLayout();
		const unsigned index_width = data_layout.getIndexSizeInBits(1);
		SmallMapVector<Value *, APInt, 4> variable_offsets;
		APInt constant_offset(index_width,0);
		if (!gep->collectOffset(data_layout,index_width,variable_offsets,
			constant_offset))
			return nullptr;
		IntegerType * index_type = IntegerType::get(module.getContext(),index_width);
		Value * address = builder.CreatePtrToInt(base,index_type,
			"openfpm_physical_base_address");
		if (!constant_offset.isZero())
			address = builder.CreateAdd(address,
				ConstantInt::get(index_type,constant_offset),
				"openfpm_physical_gep_address");
		for (const auto & [index,multiplier] : variable_offsets)
		{
			if (!index->getType()->isIntegerTy()) return nullptr;
			Value * offset = builder.CreateSExtOrTrunc(index,index_type,
				"openfpm_physical_gep_index");
			if (!multiplier.isOne())
				offset = builder.CreateMul(offset,
					ConstantInt::get(index_type,multiplier),
					"openfpm_physical_gep_offset");
			address = builder.CreateAdd(address,offset,
				"openfpm_physical_gep_address");
		}
		return builder.CreateIntToPtr(address,physical_type,
			"openfpm_physical_gep");
	}
	if (auto * selection = dyn_cast<SelectInst>(value))
	{
		dead_chain.push_back(selection);
		Value * true_pointer = rebuildPhysicalPointerControlFlow(
			selection->getTrueValue(),module,builder,rebuilt_phis,dead_chain,depth+1);
		Value * false_pointer = rebuildPhysicalPointerControlFlow(
			selection->getFalseValue(),module,builder,rebuilt_phis,dead_chain,depth+1);
		if (true_pointer == nullptr || false_pointer == nullptr) return nullptr;
		IntegerType * address_type = IntegerType::get(module.getContext(),
			module.getDataLayout().getIndexSizeInBits(1));
		Value * true_address = builder.CreatePtrToInt(true_pointer,address_type,
			"openfpm_physical_true_address");
		Value * false_address = builder.CreatePtrToInt(false_pointer,address_type,
			"openfpm_physical_false_address");
		Value * address = builder.CreateSelect(selection->getCondition(),
			true_address,false_address,"openfpm_physical_address_select");
		return builder.CreateIntToPtr(address,physical_type,
			"openfpm_physical_pointer");
	}
	if (auto * phi = dyn_cast<PHINode>(value))
	{
		if (auto found = rebuilt_phis.find(phi); found != rebuilt_phis.end())
			return builder.CreateIntToPtr(found->second,physical_type,
				"openfpm_physical_pointer");
		if (!isPhysicalLoopPhi(phi)) return nullptr;
		IntegerType * address_type = IntegerType::get(module.getContext(),
			module.getDataLayout().getIndexSizeInBits(1));
		PHINode * replacement = PHINode::Create(address_type,
			phi->getNumIncomingValues(),"openfpm_physical_address_phi",
			phi->getIterator());
		rebuilt_phis.emplace(phi,replacement);
		dead_chain.push_back(phi);
		for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
		{
			BasicBlock * incoming_block = phi->getIncomingBlock(i);
			IRBuilder<> incoming_builder(incoming_block->getTerminator());
			incoming_builder.SetCurrentDebugLocation(phi->getDebugLoc());
			Value * incoming = rebuildPhysicalPointerControlFlow(
				phi->getIncomingValue(i),module,incoming_builder,rebuilt_phis,
				dead_chain,depth+1);
			if (incoming == nullptr) return nullptr;
			Value * incoming_address = incoming_builder.CreatePtrToInt(incoming,
				address_type,"openfpm_physical_incoming_address");
			replacement->addIncoming(incoming_address,incoming_block);
		}
		return builder.CreateIntToPtr(replacement,physical_type,
			"openfpm_physical_pointer");
	}
	if (isa<ConstantPointerNull>(value))
		return ConstantPointerNull::get(physical_type);
	return nullptr;
}

// Rebuild the terminal AS1 pointer from its integer device address and the
// collected byte offset.  The load/store/atomic supplies the pointee type to
// clspv.  In particular, do not leave a typed GEP rooted at clang's original
// `inttoptr`: InstCombine can legally canonicalize that GEP back to the
// original byte-array GEP, which makes clspv emit aggregate uchar copies that
// MoltenVK cannot translate to MSL.
[[nodiscard]] Value * recoverTypedPhysicalPointer(Value * value,
	Type * access_type, Module & module, IRBuilder<> & builder,
	SmallVectorImpl<Instruction *> & dead_chain)
{
	const DataLayout & data_layout = module.getDataLayout();
	const unsigned index_width = data_layout.getIndexSizeInBits(1);
	SmallMapVector<Value *, APInt, 4> variable_offsets;
	APInt constant_offset(index_width, 0);
	Value * root = nullptr;
	bool crossed_generic_cast = false;
	if (!collectPhysicalPointerPath(value, data_layout, index_width,
		variable_offsets, constant_offset, dead_chain, root,
		crossed_generic_cast) || root == nullptr)
		return nullptr;

	std::uint64_t access_size = 0;
	if (!access_type->isSized() ||
		!fixedSize(data_layout.getTypeAllocSize(access_type), access_size,
			"physical memory access") || access_size == 0)
		return nullptr;

	IntegerType * index_type = IntegerType::get(module.getContext(), index_width);
	Value * byte_offset = ConstantInt::get(index_type, constant_offset);
	for (const auto & [index, multiplier] : variable_offsets)
	{
		if (!index->getType()->isIntegerTy()) return nullptr;
		Value * normalized_index = builder.CreateSExtOrTrunc(index, index_type,
			"openfpm_index");
		if (!multiplier.isOne())
			normalized_index = builder.CreateMul(normalized_index,
				ConstantInt::get(index_type, multiplier),
				"openfpm_scaled_index");
		byte_offset = builder.CreateAdd(byte_offset, normalized_index,
			"openfpm_byte_offset");
	}

	Value * base_address = nullptr;
	if (auto * int_to_ptr = dyn_cast<IntToPtrInst>(root))
	{
		base_address = int_to_ptr->getOperand(0);
		if (!base_address->getType()->isIntegerTy()) return nullptr;
		base_address = builder.CreateZExtOrTrunc(base_address, index_type,
			"openfpm_base_address");
	}
	else if (auto * selection = dyn_cast<SelectInst>(root))
	{
		SmallVector<Instruction *, 8> true_chain;
		Value * true_pointer = recoverTypedPhysicalPointer(
			selection->getTrueValue(),access_type,module,builder,true_chain);
		SmallVector<Instruction *, 8> false_chain;
		Value * false_pointer = recoverTypedPhysicalPointer(
			selection->getFalseValue(),access_type,module,builder,false_chain);
		if (true_pointer == nullptr || false_pointer == nullptr) return nullptr;
		for (Instruction * instruction : true_chain)
			dead_chain.push_back(instruction);
		for (Instruction * instruction : false_chain)
			dead_chain.push_back(instruction);
		dead_chain.push_back(selection);
		Value * selected_pointer = builder.CreateSelect(selection->getCondition(),
			true_pointer,false_pointer,"openfpm_selected_pointer");
		base_address = builder.CreatePtrToInt(selected_pointer,index_type,
			"openfpm_base_address");
	}
	else if (isa<ConstantPointerNull>(root))
	{
		base_address = ConstantInt::get(index_type,0);
	}
	else if (auto * phi = dyn_cast<PHINode>(root))
	{
		std::map<PHINode *, PHINode *> rebuilt_phis;
		Value * physical = rebuildPhysicalPointerControlFlow(phi,module,builder,
			rebuilt_phis,dead_chain);
		if (physical == nullptr) return nullptr;
		base_address = builder.CreatePtrToInt(physical,index_type,
			"openfpm_base_address");
	}
	else
	{
		base_address = builder.CreatePtrToInt(root, index_type,
			"openfpm_base_address");
	}

	Value * address = base_address;
	if (auto * constant = dyn_cast<ConstantInt>(byte_offset);
		constant == nullptr || !constant->isZero())
		address = builder.CreateAdd(base_address, byte_offset,
			"openfpm_physical_address");
	return builder.CreateIntToPtr(address,
		PointerType::get(module.getContext(), 1), "openfpm_typed_pointer");
}

void eraseDeadGenericPointerChain(ArrayRef<Instruction *> chain)
{
	for (Instruction * instruction : chain)
	{
		if (instruction->use_empty() &&
			(isa<GetElementPtrInst>(instruction) ||
			 isa<BitCastInst>(instruction) ||
			 isa<AddrSpaceCastInst>(instruction) ||
			 isa<SelectInst>(instruction) ||
			 isa<FreezeInst>(instruction) ||
			 isa<PHINode>(instruction) ||
			 isa<IntToPtrInst>(instruction) ||
			 isa<ExtractValueInst>(instruction) ||
			 isa<InsertValueInst>(instruction)))
			instruction->eraseFromParent();
	}
}

// Once physical memory operations have been rebuilt, fold the remaining
// scalar extractvalue users of loadPackedValue's insertvalue chain. This is a
// narrow aggregate cleanup which avoids clspv seeing dead generic pointers;
// unlike a broad instcombine pass it does not rewrite primitive control flow.
void foldInsertedAggregateExtractions(Module & module)
{
	SmallVector<ExtractValueInst *, 32> extractions;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * extraction = dyn_cast<ExtractValueInst>(&instruction))
					extractions.push_back(extraction);

	for (ExtractValueInst * extraction : extractions)
	{
		SmallVector<Instruction *, 16> insertion_chain;
		Value * inserted = findInsertedAggregateValue(
			extraction->getAggregateOperand(),extraction->getIndices(),
			insertion_chain);
		if (inserted == nullptr || inserted->getType() != extraction->getType())
			continue;
		extraction->replaceAllUsesWith(inserted);
		extraction->eraseFromParent();
		for (Instruction * instruction : insertion_chain)
			if (instruction->use_empty()) instruction->eraseFromParent();
	}

	bool removed = false;
	do
	{
		removed = false;
		SmallVector<Instruction *, 32> dead_values;
		for (Function & function : module)
			for (BasicBlock & block : function)
				for (Instruction & instruction : block)
					if (instruction.use_empty() &&
						(isa<InsertValueInst>(instruction) ||
						 isa<IntToPtrInst>(instruction)))
						dead_values.push_back(&instruction);
		for (Instruction * instruction : dead_values)
		{
			instruction->eraseFromParent();
			removed = true;
		}
	}
	while (removed);
}

void lowerChipStarAtomicAdds(Module & module)
{
	SmallVector<CallInst *, 16> calls;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * call = dyn_cast<CallInst>(&instruction))
					if (Function * callee = call->getCalledFunction();
						callee != nullptr &&
						(isChipStarIntegerAtomicAdd(callee->getName()) ||
						 isChipStarFloatAtomicAdd(callee->getName())))
						calls.push_back(call);

	const SyncScope::ID device_scope =
		module.getContext().getOrInsertSyncScopeID("device");
	for (CallInst * call : calls)
	{
		Function * callee = call->getCalledFunction();
		if (callee != nullptr && isChipStarFloatAtomicAdd(callee->getName()))
		{
			if (call->arg_size() != 2 || !call->getType()->isFloatTy() ||
				!call->getArgOperand(0)->getType()->isIntegerTy() ||
				call->getArgOperand(1)->getType() != call->getType())
			{
				errs() << "Unsupported chipStar float atomic-add signature: ";
				call->print(errs());
				errs() << '\n';
				continue;
			}

			BasicBlock * before = call->getParent();
			Function * function = before->getParent();
			BasicBlock * continuation = before->splitBasicBlock(
				call->getIterator(), "openfpm_atomic_float_done");
			before->getTerminator()->eraseFromParent();
			BasicBlock * loop = BasicBlock::Create(module.getContext(),
				"openfpm_atomic_float_loop", function, continuation);

			IRBuilder<> pre_builder(before);
			pre_builder.SetCurrentDebugLocation(call->getDebugLoc());
			IntegerType * address_type = IntegerType::get(module.getContext(),
				module.getDataLayout().getPointerSizeInBits(1));
			Value * address = pre_builder.CreateZExtOrTrunc(
				call->getArgOperand(0), address_type,
				"openfpm_atomic_float_address");
			Value * pointer = pre_builder.CreateIntToPtr(address,
				PointerType::get(module.getContext(), 1),
				"openfpm_atomic_float_pointer");
			Type * integer_type = Type::getInt32Ty(module.getContext());
			FunctionType * compare_exchange_type = FunctionType::get(integer_type,
				{pointer->getType(), integer_type, integer_type}, false);
			FunctionCallee compare_exchange = module.getOrInsertFunction(
				"_Z14atomic_cmpxchgPU3AS1Vjjj", compare_exchange_type);
			if (Function * compare_exchange_function =
				dyn_cast<Function>(compare_exchange.getCallee()))
			{
				compare_exchange_function->setCallingConv(CallingConv::SPIR_FUNC);
				compare_exchange_function->addFnAttr(Attribute::Convergent);
			}
			Value * zero = ConstantInt::get(cast<IntegerType>(integer_type), 0);
			CallInst * initial = pre_builder.CreateCall(compare_exchange_type,
				compare_exchange.getCallee(), {pointer, zero, zero},
				"openfpm_atomic_float_initial");
			initial->setCallingConv(CallingConv::SPIR_FUNC);
			pre_builder.CreateBr(loop);

			IRBuilder<> loop_builder(loop);
			loop_builder.SetCurrentDebugLocation(call->getDebugLoc());
			PHINode * expected = loop_builder.CreatePHI(initial->getType(), 2,
				"openfpm_atomic_float_expected");
			expected->addIncoming(initial, before);
			Value * old_float = loop_builder.CreateBitCast(expected,
				call->getType(), "openfpm_atomic_float_old");
			Instruction * sum = cast<Instruction>(loop_builder.CreateFAdd(
				old_float, call->getArgOperand(1), "openfpm_atomic_float_sum"));
			sum->setFastMathFlags(call->getFastMathFlags());
			Value * desired = loop_builder.CreateBitCast(sum, initial->getType(),
				"openfpm_atomic_float_desired");
			CallInst * observed = loop_builder.CreateCall(compare_exchange_type,
				compare_exchange.getCallee(), {pointer, expected, desired},
				"openfpm_atomic_float_observed");
			observed->setCallingConv(CallingConv::SPIR_FUNC);
			Value * success = loop_builder.CreateICmpEQ(observed, expected,
				"openfpm_atomic_float_success");
			expected->addIncoming(observed, loop);
			loop_builder.CreateCondBr(success, continuation, loop);

			call->replaceAllUsesWith(old_float);
			call->eraseFromParent();
			continue;
		}
		if (call->arg_size() != 2 || !call->getType()->isIntegerTy() ||
			call->getType() != call->getArgOperand(1)->getType() ||
			(call->getType()->getIntegerBitWidth() != 32 &&
			 call->getType()->getIntegerBitWidth() != 64))
		{
			errs() << "Unsupported chipStar atomic-add signature: ";
			call->print(errs());
			errs() << '\n';
			continue;
		}

		IRBuilder<> builder(call);
		builder.SetCurrentDebugLocation(call->getDebugLoc());
		SmallVector<Instruction *, 8> dead_chain;
		Value * pointer = recoverTypedPhysicalPointer(call->getArgOperand(0),
			call->getType(), module, builder, dead_chain);
		if (pointer == nullptr) continue;

		AtomicRMWInst * atomic = builder.CreateAtomicRMW(AtomicRMWInst::Add,
			pointer, call->getArgOperand(1),
			module.getDataLayout().getABITypeAlign(call->getType()),
			AtomicOrdering::Monotonic, device_scope);
		atomic->takeName(call);
		call->replaceAllUsesWith(atomic);
		call->eraseFromParent();
		eraseDeadGenericPointerChain(dead_chain);
	}
}

// When the first ABI pass cannot yet recover a pointer buried in a by-value
// aggregate, the mandatory inline/SROA pipeline exposes chipStar's OpenCL
// atomic helper instead.  Turn that exact relaxed, device-scope integer add
// back into LLVM atomic IR before rebuilding its physical pointer below.
void lowerInlinedChipStarAtomicAdds(Module & module)
{
	SmallVector<CallInst *, 16> calls;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * call = dyn_cast<CallInst>(&instruction))
					if (Function * callee = call->getCalledFunction();
						callee != nullptr &&
						callee->getName().starts_with(
							"_Z25atomic_fetch_add_explicit"))
						calls.push_back(call);

	const SyncScope::ID device_scope =
		module.getContext().getOrInsertSyncScopeID("device");
	for (CallInst * call : calls)
	{
		if (call->arg_size() != 4 || !call->getType()->isIntegerTy() ||
			call->getType() != call->getArgOperand(1)->getType() ||
			(call->getType()->getIntegerBitWidth() != 32 &&
			 call->getType()->getIntegerBitWidth() != 64) ||
			!call->getArgOperand(0)->getType()->isPointerTy())
			continue;

		auto * ordering = dyn_cast<ConstantInt>(call->getArgOperand(2));
		auto * scope = dyn_cast<ConstantInt>(call->getArgOperand(3));
		if (ordering == nullptr || !ordering->isZero() || scope == nullptr ||
			!scope->equalsInt(2))
			continue;

		IRBuilder<> builder(call);
		builder.SetCurrentDebugLocation(call->getDebugLoc());
		AtomicRMWInst * atomic = builder.CreateAtomicRMW(AtomicRMWInst::Add,
			call->getArgOperand(0), call->getArgOperand(1),
			module.getDataLayout().getABITypeAlign(call->getType()),
			AtomicOrdering::Monotonic, device_scope);
		atomic->takeName(call);
		call->replaceAllUsesWith(atomic);
		call->eraseFromParent();
	}
}

// clspv's address-space lowering cannot represent a load/store whose payload
// is itself a generic pointer and whose address is a PhysicalStorageBuffer
// pointer.  Device pointers are plain pointer-width addresses in OpenFPM's
// packed ABI, so transfer them as integers at the physical-memory boundary.
// Loads are rewritten first because a following pointer store may consume the
// generic pointer reconstructed by that load.
void lowerPointerValuedPhysicalMemoryAccesses(Module & module)
{
	SmallVector<LoadInst *, 16> loads;
	SmallVector<StoreInst *, 16> stores;
	for (Function & function : module)
	{
		for (BasicBlock & block : function)
		{
			for (Instruction & instruction : block)
			{
				if (auto * load = dyn_cast<LoadInst>(&instruction);
					load != nullptr && load->getType()->isPointerTy() &&
					load->getPointerAddressSpace() == 4)
					loads.push_back(load);
				else if (auto * store = dyn_cast<StoreInst>(&instruction);
					store != nullptr &&
					store->getValueOperand()->getType()->isPointerTy() &&
					store->getPointerAddressSpace() == 4)
					stores.push_back(store);
			}
		}
	}

	const DataLayout & data_layout = module.getDataLayout();
	IntegerType * address_type = IntegerType::get(module.getContext(),
		data_layout.getPointerSizeInBits(1));
	Type * byte_type = Type::getInt8Ty(module.getContext());
	PointerType * physical_pointer_type =
		PointerType::get(module.getContext(), 1);

	for (LoadInst * load : loads)
	{
		if (load->getType()->getPointerAddressSpace() != 4) continue;
		IRBuilder<> builder(load);
		builder.SetCurrentDebugLocation(load->getDebugLoc());
		SmallVector<Instruction *, 8> dead_chain;
		Value * address = recoverTypedPhysicalPointer(load->getPointerOperand(),
			address_type, module, builder, dead_chain);
		if (address == nullptr) continue;

		LoadInst * integer_load = builder.CreateAlignedLoad(address_type, address,
			load->getAlign(), load->getName() + ".address");
		integer_load->setVolatile(load->isVolatile());
		if (load->isAtomic())
			integer_load->setAtomic(load->getOrdering(), load->getSyncScopeID());
		integer_load->copyMetadata(*load);
		Value * physical = builder.CreateIntToPtr(integer_load,
			physical_pointer_type, load->getName() + ".physical");
		Value * generic = builder.CreateAddrSpaceCast(physical, load->getType(),
			load->getName() + ".generic");
		load->replaceAllUsesWith(generic);
		load->eraseFromParent();
		eraseDeadGenericPointerChain(dead_chain);
	}

	for (StoreInst * store : stores)
	{
		IRBuilder<> builder(store);
		builder.SetCurrentDebugLocation(store->getDebugLoc());
		SmallVector<Instruction *, 16> dead_chain;
		Value * destination = recoverTypedPhysicalPointer(
			store->getPointerOperand(), address_type, module, builder, dead_chain);
		if (destination == nullptr) continue;

		Value * pointer_value = store->getValueOperand();
		Value * integer_value = nullptr;
		if (isa<ConstantPointerNull>(pointer_value))
			integer_value = ConstantInt::get(address_type, 0);
		else
		{
			SmallVector<Instruction *, 8> value_chain;
			Value * physical = recoverTypedPhysicalPointer(pointer_value, byte_type,
				module, builder, value_chain);
			if (physical == nullptr) continue;
			for (Instruction * instruction : value_chain)
				if (std::find(dead_chain.begin(), dead_chain.end(), instruction) ==
					dead_chain.end())
					dead_chain.push_back(instruction);
			integer_value = builder.CreatePtrToInt(physical, address_type,
				"openfpm_stored_pointer_address");
		}

		StoreInst * integer_store = builder.CreateAlignedStore(integer_value,
			destination, store->getAlign());
		integer_store->setVolatile(store->isVolatile());
		if (store->isAtomic())
			integer_store->setAtomic(store->getOrdering(), store->getSyncScopeID());
		integer_store->copyMetadata(*store);
		store->eraseFromParent();
		eraseDeadGenericPointerChain(dead_chain);
	}
}

// LLVM permits pointers to be stored in aggregates, but SPIR-V logical
// storage classes do not permit arbitrary pointer-valued private memory.  HIP
// uses this pattern for by-value `arr_ptr<N>` arguments: the wrapper creates a
// private array of device pointers and the kernel indexes it dynamically.
// Preserve the ordinary C++ object representation while exposing those leaf
// values to clspv as pointer-width integers.  This is an ABI transformation,
// not a kernel-specific source rewrite.
void lowerPointerValuedMemoryPayloads(Module & module)
{
	SmallVector<LoadInst *, 32> loads;
	SmallVector<StoreInst *, 32> stores;
	SmallVector<AllocaInst *, 16> allocations;
	for (Function & function : module)
	{
		for (BasicBlock & block : function)
		{
			for (Instruction & instruction : block)
			{
				if (auto * load = dyn_cast<LoadInst>(&instruction);
					load != nullptr && load->getType()->isPointerTy())
					loads.push_back(load);
				else if (auto * store = dyn_cast<StoreInst>(&instruction);
					store != nullptr &&
					store->getValueOperand()->getType()->isPointerTy())
					stores.push_back(store);
				else if (auto * allocation = dyn_cast<AllocaInst>(&instruction);
					allocation != nullptr &&
					containsPointerLeaf(allocation->getAllocatedType()))
					allocations.push_back(allocation);
			}
		}
	}

	const DataLayout & data_layout = module.getDataLayout();
	auto pointerIntegerType = [&](Type * pointer_type)
	{
		auto * pointer = cast<PointerType>(pointer_type);
		return IntegerType::get(module.getContext(),
			data_layout.getPointerSizeInBits(pointer->getAddressSpace()));
	};
	auto pointerAddress = [&](Value * pointer, IntegerType * integer_type,
		IRBuilder<> & builder) -> Value *
	{
		if (isa<ConstantPointerNull>(pointer))
			return ConstantInt::get(integer_type, 0);
		Value * source = pointer;
		while (auto * cast = dyn_cast<AddrSpaceCastInst>(source))
			source = cast->getOperand(0);
		if (auto * conversion = dyn_cast<IntToPtrInst>(source))
			return builder.CreateZExtOrTrunc(conversion->getOperand(0),
				integer_type, "openfpm_pointer_address");
		return builder.CreatePtrToInt(pointer, integer_type,
			"openfpm_pointer_address");
	};

	for (LoadInst * load : loads)
	{
		IntegerType * integer_type = pointerIntegerType(load->getType());
		IRBuilder<> builder(load);
		builder.SetCurrentDebugLocation(load->getDebugLoc());
		LoadInst * integer_load = builder.CreateAlignedLoad(integer_type,
			load->getPointerOperand(), load->getAlign(),
			load->getName() + ".address");
		integer_load->setVolatile(load->isVolatile());
		if (load->isAtomic())
			integer_load->setAtomic(load->getOrdering(), load->getSyncScopeID());
		integer_load->copyMetadata(*load);
		Value * pointer = builder.CreateIntToPtr(integer_load, load->getType(),
			load->getName() + ".pointer");
		load->replaceAllUsesWith(pointer);
		load->eraseFromParent();
	}

	for (StoreInst * store : stores)
	{
		Value * pointer = store->getValueOperand();
		IntegerType * integer_type = pointerIntegerType(pointer->getType());
		IRBuilder<> builder(store);
		builder.SetCurrentDebugLocation(store->getDebugLoc());
		Value * integer = pointerAddress(pointer, integer_type, builder);
		StoreInst * integer_store = builder.CreateAlignedStore(integer,
			store->getPointerOperand(), store->getAlign());
		integer_store->setVolatile(store->isVolatile());
		if (store->isAtomic())
			integer_store->setAtomic(store->getOrdering(), store->getSyncScopeID());
		integer_store->copyMetadata(*store);
		store->eraseFromParent();
	}

	for (AllocaInst * allocation : allocations)
	{
		std::uint64_t size = 0;
		if (!fixedSize(data_layout.getTypeAllocSize(
			allocation->getAllocatedType()), size,
			"pointer-containing private allocation") || size == 0)
			continue;
		IRBuilder<> builder(allocation);
		builder.SetCurrentDebugLocation(allocation->getDebugLoc());
		Type * bytes = ArrayType::get(Type::getInt8Ty(module.getContext()), size);
		AllocaInst * replacement = builder.CreateAlloca(bytes,
			allocation->getAddressSpace(), allocation->getArraySize(),
			allocation->getName() + ".bytes");
		replacement->setAlignment(allocation->getAlign());
		replacement->copyMetadata(*allocation);
		allocation->replaceAllUsesWith(replacement);
		allocation->eraseFromParent();
	}
}

// clspv's printf lowering expects its own OpenCL literal-pointer form and
// crashes on HIP's physical-to-generic constant-expression cast.  MoltenVK
// also needs a printf buffer/runtime ABI which OpenFPM does not provide yet.
// Treat device printf as a diagnostic-only no-op at this backend boundary;
// kernel computation and control flow remain unchanged.
void lowerUnsupportedDevicePrintf(Module & module)
{
	SmallVector<CallInst *, 8> calls;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * call = dyn_cast<CallInst>(&instruction);
					call != nullptr && call->getCalledFunction() != nullptr &&
					call->getCalledFunction()->getName() == "printf")
					calls.push_back(call);

	for (CallInst * call : calls)
	{
		if (!call->getType()->isVoidTy())
			call->replaceAllUsesWith(Constant::getNullValue(call->getType()));
		call->eraseFromParent();
	}
	if (Function * printf_function = module.getFunction("printf");
		printf_function != nullptr && printf_function->use_empty())
		printf_function->eraseFromParent();

	SmallVector<GlobalVariable *, 8> unused_constants;
	for (GlobalVariable & global : module.globals())
	{
		global.removeDeadConstantUsers();
		if (global.isConstant() && global.hasLocalLinkage() && global.use_empty())
			unused_constants.push_back(&global);
	}
	for (GlobalVariable * global : unused_constants) global->eraseFromParent();
}

void lowerPhysicalMemoryAccesses(Module & module)
{
	struct MemoryAccess
	{
		Instruction * instruction = nullptr;
		Value * pointer = nullptr;
		Type * value_type = nullptr;
		unsigned pointer_operand = 0;
	};
	SmallVector<MemoryAccess, 32> accesses;
	for (Function & function : module)
	{
		for (BasicBlock & block : function)
		{
			for (Instruction & instruction : block)
			{
				MemoryAccess access;
				if (auto * load = dyn_cast<LoadInst>(&instruction))
				{
					access = {load, load->getPointerOperand(), load->getType(), 0};
				}
				else if (auto * store = dyn_cast<StoreInst>(&instruction))
				{
					access = {store, store->getPointerOperand(),
						store->getValueOperand()->getType(), 1};
				}
				else if (auto * atomic = dyn_cast<AtomicRMWInst>(&instruction))
				{
					access = {atomic, atomic->getPointerOperand(),
						atomic->getValOperand()->getType(), 0};
				}
				if (access.pointer != nullptr && !access.value_type->isPointerTy() &&
					access.pointer->getType()->isPointerTy()
					&& access.pointer->getType()->getPointerAddressSpace() == 4)
					accesses.push_back(access);
			}
		}
	}

	for (const MemoryAccess & access : accesses)
	{
		IRBuilder<> builder(access.instruction);
		builder.SetCurrentDebugLocation(access.instruction->getDebugLoc());
		SmallVector<Instruction *, 8> dead_chain;
		Value * pointer = recoverTypedPhysicalPointer(access.pointer,
			access.value_type, module, builder, dead_chain);
		if (pointer == nullptr) continue;
		access.instruction->setOperand(access.pointer_operand, pointer);
		eraseDeadGenericPointerChain(dead_chain);
	}
}

// LLVM's ptrtoaddr preserves non-address pointer metadata for targets that
// need it.  OpenFPM device pointers carry no such metadata, while clspv's
// LowerAddrSpaceCast pass does not accept ptrtoaddr on a generic pointer
// rooted in a PhysicalStorageBuffer address.  Recover that physical address
// and use an ordinary integer conversion.
void lowerPhysicalPtrToAddr(Module & module)
{
	SmallVector<PtrToAddrInst *, 16> conversions;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * conversion = dyn_cast<PtrToAddrInst>(&instruction);
					conversion != nullptr &&
					conversion->getOperand(0)->getType()->isPointerTy() &&
					conversion->getOperand(0)->getType()->getPointerAddressSpace() == 4)
					conversions.push_back(conversion);

	Type * byte_type = Type::getInt8Ty(module.getContext());
	for (PtrToAddrInst * conversion : conversions)
	{
		IRBuilder<> builder(conversion);
		builder.SetCurrentDebugLocation(conversion->getDebugLoc());
		SmallVector<Instruction *, 8> dead_chain;
		Value * physical = recoverTypedPhysicalPointer(conversion->getOperand(0),
			byte_type, module, builder, dead_chain);
		// Generic-pointer function arguments and loop-carried phis do not expose
		// a recoverable local AS1 chain.  In OpenFPM's physical pointer ABI the
		// generic pointer still contains exactly the integer device address, so
		// ptrtoint is the metadata-free equivalent of ptrtoaddr.
		Value * pointer = physical != nullptr ? physical : conversion->getOperand(0);
		Value * integer = builder.CreatePtrToInt(pointer, conversion->getType(),
			"openfpm_pointer_address");
		conversion->replaceAllUsesWith(integer);
		conversion->eraseFromParent();
		eraseDeadGenericPointerChain(dead_chain);
	}
}

// Earlier ABI stages may already have expressed a C++ pointer comparison or
// iterator subtraction as ptrtoint while the pointer was still generic.  Once
// aggregate pointer loads are exposed, move those conversions to the rebuilt
// AS1 value so clspv never has to infer through a generic pointer phi.
void lowerPhysicalPtrToInt(Module & module)
{
	SmallVector<PtrToIntInst *, 16> conversions;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * conversion = dyn_cast<PtrToIntInst>(&instruction);
					conversion != nullptr &&
					conversion->getOperand(0)->getType()->getPointerAddressSpace() == 4)
					conversions.push_back(conversion);

	Type * byte_type = Type::getInt8Ty(module.getContext());
	for (PtrToIntInst * conversion : conversions)
	{
		IRBuilder<> builder(conversion);
		builder.SetCurrentDebugLocation(conversion->getDebugLoc());
		SmallVector<Instruction *, 16> dead_chain;
		Value * physical = recoverTypedPhysicalPointer(conversion->getOperand(0),
			byte_type,module,builder,dead_chain);
		if (physical == nullptr) continue;
		Value * integer = builder.CreatePtrToInt(physical,conversion->getType(),
			"openfpm_pointer_address");
		conversion->replaceAllUsesWith(integer);
		conversion->eraseFromParent();
		eraseDeadGenericPointerChain(dead_chain);
	}
}

// SPIR-V's PhysicalStorageBuffer pointers cannot be operands of logical pointer
// comparison instructions in the Vulkan environment. HIP C++ emits both
// equality and ordered comparisons on generic (AS4) pointers: STL-style end
// checks use equality, while packed-buffer iterators use `<`/`>=`. Recover the
// underlying AS1 physical addresses using the same path analysis as
// loads/stores, then apply the original predicate to their integer device
// addresses. Keeping this in the ABI pass preserves the source-level HIP
// semantics without requiring Metal-specific kernel changes.
void lowerPhysicalPointerComparisons(Module & module)
{
	SmallVector<ICmpInst *, 16> comparisons;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * comparison = dyn_cast<ICmpInst>(&instruction);
					comparison != nullptr &&
					comparison->getOperand(0)->getType()->isPointerTy() &&
					(comparison->getOperand(0)->getType()->getPointerAddressSpace() == 1 ||
					 comparison->getOperand(0)->getType()->getPointerAddressSpace() == 4))
					comparisons.push_back(comparison);

	const DataLayout & data_layout = module.getDataLayout();
	IntegerType * address_type = IntegerType::get(module.getContext(),
		data_layout.getPointerSizeInBits(1));
	Type * byte_type = Type::getInt8Ty(module.getContext());
	for (ICmpInst * comparison : comparisons)
	{
		IRBuilder<> builder(comparison);
		builder.SetCurrentDebugLocation(comparison->getDebugLoc());
		SmallVector<Instruction *, 16> dead_chain;
		Value * left = recoverTypedPhysicalPointer(comparison->getOperand(0),
			byte_type, module, builder, dead_chain);
		SmallVector<Instruction *, 8> right_chain;
		Value * right = recoverTypedPhysicalPointer(comparison->getOperand(1),
			byte_type, module, builder, right_chain);
		for (Instruction * instruction : right_chain)
			if (std::find(dead_chain.begin(), dead_chain.end(), instruction) ==
				dead_chain.end())
				dead_chain.push_back(instruction);

		Value * left_pointer = left != nullptr ? left : comparison->getOperand(0);
		Value * right_pointer = right != nullptr ? right : comparison->getOperand(1);
		Value * left_address = builder.CreatePtrToInt(left_pointer, address_type,
			"openfpm_compare_left_address");
		Value * right_address = builder.CreatePtrToInt(right_pointer, address_type,
			"openfpm_compare_right_address");
		// Keep InstCombine from canonicalizing the integer equality back into an
		// invalid PhysicalStorageBuffer pointer comparison.
		left_address = builder.CreateFreeze(left_address,
			"openfpm_compare_left_address_frozen");
		right_address = builder.CreateFreeze(right_address,
			"openfpm_compare_right_address_frozen");
		Value * integer_comparison = builder.CreateICmp(
			comparison->getPredicate(), left_address, right_address,
			"openfpm_pointer_comparison");
		comparison->replaceAllUsesWith(integer_comparison);
		comparison->eraseFromParent();
		eraseDeadGenericPointerChain(dead_chain);
	}
}

// Metal on Apple GPUs has no double-precision arithmetic, while ordinary C++
// promotions can introduce a double expression into an otherwise float-only
// HIP kernel.  The especially common template-dependent form
//
//     float_value += 0.05;
//
// reaches LLVM as fptrunc(fadd(fpext(float_value), double 0.05)).  This is not
// an explicit fp64 data path: every non-constant leaf is still a float and the
// result is immediately converted back to float.  Rebuild such closed
// expression trees in the float domain so existing HIP/CUDA sources retain
// their intended float-first device behaviour on Metal.  Integral conversions
// are also safe leaves when the only externally visible result is float; this
// covers expressions such as `float_value = 1000.0 + thread_index`. Expressions
// fed by a double argument, load, call, or phi are deliberately left alone;
// real fp64 remains unsupported rather than being silently narrowed.
[[nodiscard]] bool canLowerPromotedFloatArithmetic(Value * value,
	SmallPtrSetImpl<Value *> & active)
{
	if (auto * extension = dyn_cast<FPExtInst>(value))
		return extension->getSrcTy()->isFloatTy() &&
			extension->getDestTy()->isDoubleTy();
	if (auto * conversion = dyn_cast<SIToFPInst>(value))
		return conversion->getSrcTy()->isIntegerTy() &&
			conversion->getDestTy()->isDoubleTy();
	if (auto * conversion = dyn_cast<UIToFPInst>(value))
		return conversion->getSrcTy()->isIntegerTy() &&
			conversion->getDestTy()->isDoubleTy();
	if (auto * constant = dyn_cast<ConstantFP>(value))
		return constant->getType()->isDoubleTy();
	if (!value->getType()->isDoubleTy() || !active.insert(value).second)
		return false;

	bool supported = false;
	if (auto * binary = dyn_cast<BinaryOperator>(value))
	{
		switch (binary->getOpcode())
		{
		case Instruction::FAdd:
		case Instruction::FSub:
		case Instruction::FMul:
		case Instruction::FDiv:
		case Instruction::FRem:
			supported = canLowerPromotedFloatArithmetic(binary->getOperand(0),active) &&
				canLowerPromotedFloatArithmetic(binary->getOperand(1),active);
			break;
		default:
			break;
		}
	}
	else if (auto * unary = dyn_cast<UnaryOperator>(value))
	{
		supported = unary->getOpcode() == Instruction::FNeg &&
			canLowerPromotedFloatArithmetic(unary->getOperand(0),active);
	}
	else if (auto * select = dyn_cast<SelectInst>(value))
	{
		supported = canLowerPromotedFloatArithmetic(select->getTrueValue(),active) &&
			canLowerPromotedFloatArithmetic(select->getFalseValue(),active);
	}
	else if (auto * freeze = dyn_cast<FreezeInst>(value))
	{
		supported = canLowerPromotedFloatArithmetic(freeze->getOperand(0),active);
	}

	active.erase(value);
	return supported;
}

void copyFloatingInstructionProperties(Value * replacement,
	const Instruction & original)
{
	auto * instruction = dyn_cast<Instruction>(replacement);
	if (instruction == nullptr) return;
	instruction->setDebugLoc(original.getDebugLoc());
	if (isa<FPMathOperator>(&original))
		instruction->copyFastMathFlags(&original);
}

Value * lowerPromotedFloatArithmeticValue(Value * value, IRBuilder<> & builder)
{
	if (auto * extension = dyn_cast<FPExtInst>(value))
		return extension->getOperand(0);
	if (auto * conversion = dyn_cast<SIToFPInst>(value))
	{
		Value * replacement = builder.CreateSIToFP(conversion->getOperand(0),
			Type::getFloatTy(value->getContext()),"openfpm_float_conversion");
		copyFloatingInstructionProperties(replacement,*conversion);
		return replacement;
	}
	if (auto * conversion = dyn_cast<UIToFPInst>(value))
	{
		Value * replacement = builder.CreateUIToFP(conversion->getOperand(0),
			Type::getFloatTy(value->getContext()),"openfpm_float_conversion");
		copyFloatingInstructionProperties(replacement,*conversion);
		return replacement;
	}
	if (auto * constant = dyn_cast<ConstantFP>(value))
	{
		APFloat converted = constant->getValueAPF();
		bool loses_information = false;
		converted.convert(APFloat::IEEEsingle(),APFloat::rmNearestTiesToEven,
			&loses_information);
		return ConstantFP::get(Type::getFloatTy(value->getContext()),converted);
	}
	if (auto * binary = dyn_cast<BinaryOperator>(value))
	{
		Value * left = lowerPromotedFloatArithmeticValue(binary->getOperand(0),builder);
		Value * right = lowerPromotedFloatArithmeticValue(binary->getOperand(1),builder);
		Value * replacement = builder.CreateBinOp(
			static_cast<Instruction::BinaryOps>(binary->getOpcode()),left,right,
			"openfpm_float_arithmetic");
		copyFloatingInstructionProperties(replacement,*binary);
		return replacement;
	}
	if (auto * unary = dyn_cast<UnaryOperator>(value))
	{
		Value * replacement = builder.CreateFNeg(
			lowerPromotedFloatArithmeticValue(unary->getOperand(0),builder),
			"openfpm_float_arithmetic");
		copyFloatingInstructionProperties(replacement,*unary);
		return replacement;
	}
	if (auto * select = dyn_cast<SelectInst>(value))
	{
		Value * replacement = builder.CreateSelect(select->getCondition(),
			lowerPromotedFloatArithmeticValue(select->getTrueValue(),builder),
			lowerPromotedFloatArithmeticValue(select->getFalseValue(),builder),
			"openfpm_float_arithmetic");
		copyFloatingInstructionProperties(replacement,*select);
		return replacement;
	}
	auto * freeze = cast<FreezeInst>(value);
	Value * replacement = builder.CreateFreeze(
		lowerPromotedFloatArithmeticValue(freeze->getOperand(0),builder),
		"openfpm_float_arithmetic");
	copyFloatingInstructionProperties(replacement,*freeze);
	return replacement;
}

void lowerPromotedFloatArithmetic(Module & module)
{
	SmallVector<FPTruncInst *, 32> truncations;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * truncation = dyn_cast<FPTruncInst>(&instruction);
					truncation != nullptr && truncation->getSrcTy()->isDoubleTy() &&
					truncation->getDestTy()->isFloatTy())
					truncations.push_back(truncation);

	for (FPTruncInst * truncation : truncations)
	{
		SmallPtrSet<Value *, 16> active;
		if (!canLowerPromotedFloatArithmetic(truncation->getOperand(0),active))
			continue;
		IRBuilder<> builder(truncation);
		builder.SetCurrentDebugLocation(truncation->getDebugLoc());
		Value * replacement = lowerPromotedFloatArithmeticValue(
			truncation->getOperand(0),builder);
		truncation->replaceAllUsesWith(replacement);
		truncation->eraseFromParent();
	}
}

// Metal lacks fp64 arithmetic, but OpenFPM legitimately moves opaque payloads
// containing doubles (for example Point<double>) in generic copy kernels.  A
// double value whose only consumers are stores is not floating-point
// computation at all.  Transfer those eight bytes as i64 so SPIR-V/Metal do
// not request the Float64 capability, while preserving the payload bit-for-bit.
// Loads participating in arithmetic, comparisons, conversions, calls, or
// control flow are intentionally untouched and continue to fail as unsupported
// fp64 instead of being silently reinterpreted.
void lowerStorageOnlyDoubleTransfers(Module & module)
{
	SmallVector<LoadInst *, 32> loads;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * load = dyn_cast<LoadInst>(&instruction);
					load != nullptr && load->getType()->isDoubleTy() &&
					!load->isAtomic())
					loads.push_back(load);

	Type * bits_type = Type::getInt64Ty(module.getContext());
	for (LoadInst * load : loads)
	{
		SmallVector<StoreInst *, 4> stores;
		bool storage_only = !load->user_empty();
		for (User * user : load->users())
		{
			auto * store = dyn_cast<StoreInst>(user);
			if (store == nullptr || store->getValueOperand() != load ||
				store->isAtomic())
			{
				storage_only = false;
				break;
			}
			stores.push_back(store);
		}
		if (!storage_only) continue;

		IRBuilder<> load_builder(load);
		load_builder.SetCurrentDebugLocation(load->getDebugLoc());
		auto * bits = load_builder.CreateLoad(bits_type,load->getPointerOperand(),
			"openfpm_fp64_storage_bits");
		bits->setAlignment(load->getAlign());
		bits->setVolatile(load->isVolatile());
		bits->copyMetadata(*load);

		for (StoreInst * store : stores)
		{
			IRBuilder<> store_builder(store);
			store_builder.SetCurrentDebugLocation(store->getDebugLoc());
			auto * replacement = store_builder.CreateStore(bits,
				store->getPointerOperand(),store->isVolatile());
			replacement->setAlignment(store->getAlign());
			replacement->copyMetadata(*store);
			store->eraseFromParent();
		}
		load->eraseFromParent();
	}
}

// Apple GPUs do not expose 64-bit floating point through Metal.  A float HIP
// kernel can nevertheless acquire an otherwise unnecessary double operation
// through the usual C++ arithmetic conversions, for example
//
//     float distance = ...;
//     if (distance > 0.000001) ...;
//
// Clang represents this as fcmp(fpext(float), double-constant).  The comparison
// can be evaluated exactly in the original float domain because the variable
// has only float values.  Round the constant toward the comparison boundary so
// this remains true even when the source double is not exactly representable as
// a float.  This is an ABI translation rule, not a source workaround: the same
// HIP/CUDA kernel stays valid and unchanged for every backend.
void lowerPromotedFloatComparisons(Module & module)
{
	SmallVector<FCmpInst *, 16> comparisons;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * comparison = dyn_cast<FCmpInst>(&instruction);
					comparison != nullptr && comparison->getType()->isIntegerTy(1))
					comparisons.push_back(comparison);

	for (FCmpInst * comparison : comparisons)
	{
		Value * extended_value = comparison->getOperand(0);
		ConstantFP * constant = dyn_cast<ConstantFP>(comparison->getOperand(1));
		CmpInst::Predicate predicate = comparison->getPredicate();
		if (constant == nullptr)
		{
			constant = dyn_cast<ConstantFP>(comparison->getOperand(0));
			extended_value = comparison->getOperand(1);
			if (constant != nullptr)
				predicate = CmpInst::getSwappedPredicate(predicate);
		}

		auto * extension = dyn_cast<FPExtInst>(extended_value);
		if (extension == nullptr || constant == nullptr ||
			!extension->getSrcTy()->isFloatTy() ||
			!extension->getDestTy()->isDoubleTy() ||
			!constant->getType()->isDoubleTy() ||
			!constant->getValueAPF().isFinite())
			continue;

		Value * source = extension->getOperand(0);
		APFloat lower = constant->getValueAPF();
		APFloat upper = constant->getValueAPF();
		bool lower_loses_information = false;
		bool upper_loses_information = false;
		lower.convert(APFloat::IEEEsingle(), APFloat::rmTowardNegative,
			&lower_loses_information);
		upper.convert(APFloat::IEEEsingle(), APFloat::rmTowardPositive,
			&upper_loses_information);
		const bool exactly_representable = !lower_loses_information &&
			!upper_loses_information;

		IRBuilder<> builder(comparison);
		builder.SetCurrentDebugLocation(comparison->getDebugLoc());
		Value * replacement = nullptr;
		if (exactly_representable)
		{
			replacement = builder.CreateFCmp(predicate, source,
				ConstantFP::get(source->getType(), lower),
				"openfpm_float_comparison");
		}
		else
		{
			switch (predicate)
			{
			case CmpInst::FCMP_OLT:
			case CmpInst::FCMP_OLE:
				replacement = builder.CreateFCmpOLT(source,
					ConstantFP::get(source->getType(), upper),
					"openfpm_float_comparison");
				break;
			case CmpInst::FCMP_ULT:
			case CmpInst::FCMP_ULE:
				replacement = builder.CreateFCmpULT(source,
					ConstantFP::get(source->getType(), upper),
					"openfpm_float_comparison");
				break;
			case CmpInst::FCMP_OGT:
			case CmpInst::FCMP_OGE:
				replacement = builder.CreateFCmpOGT(source,
					ConstantFP::get(source->getType(), lower),
					"openfpm_float_comparison");
				break;
			case CmpInst::FCMP_UGT:
			case CmpInst::FCMP_UGE:
				replacement = builder.CreateFCmpUGT(source,
					ConstantFP::get(source->getType(), lower),
					"openfpm_float_comparison");
				break;
			case CmpInst::FCMP_OEQ:
			case CmpInst::FCMP_FALSE:
				replacement = ConstantInt::getFalse(module.getContext());
				break;
			case CmpInst::FCMP_UNE:
			case CmpInst::FCMP_TRUE:
				replacement = ConstantInt::getTrue(module.getContext());
				break;
			case CmpInst::FCMP_ONE:
			case CmpInst::FCMP_ORD:
				replacement = builder.CreateFCmpORD(source, source,
					"openfpm_float_comparison");
				break;
			case CmpInst::FCMP_UEQ:
			case CmpInst::FCMP_UNO:
				replacement = builder.CreateFCmpUNO(source, source,
					"openfpm_float_comparison");
				break;
			default:
				break;
			}
		}

		if (replacement == nullptr) continue;
		comparison->replaceAllUsesWith(replacement);
		comparison->eraseFromParent();
		if (extension->use_empty()) extension->eraseFromParent();
	}
}

// Older SPIRV-Cross emits `reinterpret_cast<ulong>(array[index])` for an
// OpConvertPtrToU whose source is a pointer-to-array access chain.  Preserve
// the same address calculation as integer base + byte offset so the generated
// MSL casts the pointer itself, not the aggregate object it designates.
void lowerAggregateGepPtrToInt(Module & module)
{
	SmallVector<PtrToIntInst *, 16> conversions;
	for (Function & function : module)
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * conversion = dyn_cast<PtrToIntInst>(&instruction);
					conversion != nullptr &&
					isa<GetElementPtrInst>(conversion->getOperand(0)))
					conversions.push_back(conversion);

	const DataLayout & data_layout = module.getDataLayout();
	for (PtrToIntInst * conversion : conversions)
	{
		auto * gep = cast<GetElementPtrInst>(conversion->getOperand(0));
		if (!conversion->getType()->isIntegerTy()) continue;
		const unsigned width = cast<IntegerType>(conversion->getType())
			->getBitWidth();
		SmallMapVector<Value *,APInt,4> variable_offsets;
		APInt constant_offset(width,0);
		if (!gep->collectOffset(data_layout,width,variable_offsets,
			constant_offset))
			continue;

		IRBuilder<> builder(conversion);
		builder.SetCurrentDebugLocation(conversion->getDebugLoc());
		Value * address = builder.CreatePtrToInt(gep->getPointerOperand(),
			conversion->getType(),"openfpm_gep_base_address");
		if (!constant_offset.isZero())
			address = builder.CreateAdd(address,
				ConstantInt::get(cast<IntegerType>(conversion->getType()),
					constant_offset),"openfpm_gep_constant_address");
		for (const auto & [index,multiplier] : variable_offsets)
		{
			if (!index->getType()->isIntegerTy()) continue;
			Value * offset = builder.CreateSExtOrTrunc(index,
				conversion->getType(),"openfpm_gep_index");
			if (!multiplier.isOne())
				offset = builder.CreateMul(offset,
					ConstantInt::get(cast<IntegerType>(conversion->getType()),
						multiplier),"openfpm_gep_scaled_index");
			address = builder.CreateAdd(address,offset,
				"openfpm_gep_address");
		}
		conversion->replaceAllUsesWith(address);
		conversion->eraseFromParent();
		if (gep->use_empty()) gep->eraseFromParent();
	}
}

[[nodiscard]] bool isDerivedFromPackedArgument(Value * value,
	const Argument * packed_argument, unsigned depth = 0)
{
	if (value == packed_argument) return true;
	if (depth > 32) return false;
	if (auto * gep = dyn_cast<GetElementPtrInst>(value))
		return isDerivedFromPackedArgument(gep->getPointerOperand(),
			packed_argument,depth + 1);
	if (auto * cast = dyn_cast<BitCastInst>(value))
		return isDerivedFromPackedArgument(cast->getOperand(0),packed_argument,
			depth + 1);
	if (auto * cast = dyn_cast<AddrSpaceCastInst>(value))
		return isDerivedFromPackedArgument(cast->getOperand(0),packed_argument,
			depth + 1);
	return false;
}

// Keep the packed launch record bit-exact through clspv's opaque-pointer type
// inference. If a record containing both floats and a 64-bit device address is
// inferred as `device float*`, SPIRV-Cross reconstructs that address through a
// float vector; subnormal address words may then be flushed to zero by Metal.
// Load floating ABI fields as same-width integers and bitcast only the value.
// This pass runs after the final LLVM cleanup so InstCombine cannot fold the
// representation back into a floating load.
void lowerPackedArgumentFloatingLoads(Module & module)
{
	SmallVector<LoadInst *, 32> loads;
	for (Function & function : module)
	{
		if (function.getCallingConv() != CallingConv::SPIR_KERNEL ||
			!function.hasFnAttribute("openfpm-moltenvk-pointer-abi") ||
			function.arg_empty())
			continue;
		Argument * packed_argument = function.getArg(0);
		for (BasicBlock & block : function)
			for (Instruction & instruction : block)
				if (auto * load = dyn_cast<LoadInst>(&instruction);
					load != nullptr && load->getType()->isFloatingPointTy() &&
					isDerivedFromPackedArgument(load->getPointerOperand(),
						packed_argument))
					loads.push_back(load);
	}

	const DataLayout & data_layout = module.getDataLayout();
	for (LoadInst * load : loads)
	{
		std::uint64_t byte_size = 0;
		if (!fixedSize(data_layout.getTypeStoreSize(load->getType()),byte_size,
			"packed floating-point load") || byte_size == 0)
			continue;
		IntegerType * bits_type = IntegerType::get(module.getContext(),
			static_cast<unsigned>(byte_size * 8));
		IRBuilder<> builder(load);
		builder.SetCurrentDebugLocation(load->getDebugLoc());
		LoadInst * bits = builder.CreateAlignedLoad(bits_type,
			load->getPointerOperand(),load->getAlign(),
			load->getName() + ".bits");
		bits->setVolatile(load->isVolatile());
		if (load->isAtomic())
			bits->setAtomic(load->getOrdering(),load->getSyncScopeID());
		bits->copyMetadata(*load);
		Value * value = builder.CreateBitCast(bits,load->getType(),
			load->getName() + ".value");
		load->replaceAllUsesWith(value);
		load->eraseFromParent();
	}
}

void copyKernelProperties(Function & destination, const Function & source)
{
	destination.setVisibility(source.getVisibility());
	destination.setDLLStorageClass(source.getDLLStorageClass());
	destination.setDSOLocal(source.isDSOLocal());
	destination.setUnnamedAddr(source.getUnnamedAddr());
	destination.setSection(source.getSection());
	destination.setPartition(source.getPartition());
	if (source.hasComdat())
		destination.setComdat(const_cast<Comdat *>(source.getComdat()));
	if (source.hasGC()) destination.setGC(source.getGC());
	if (source.hasPersonalityFn())
		destination.setPersonalityFn(source.getPersonalityFn());

	LLVMContext & context = destination.getContext();
	AttrBuilder function_attributes(context, source.getAttributes().getFnAttrs());
	destination.addFnAttrs(function_attributes);
	AttrBuilder return_attributes(context, source.getAttributes().getRetAttrs());
	destination.addRetAttrs(return_attributes);
	destination.copyMetadata(&source, 0);
}

void markReachableDeviceFunctionsAlwaysInline(Module & module)
{
	// The per-entry internalize/globaldce step has already removed unrelated
	// definitions, so every remaining internal definition is reachable from
	// this exact kernel.  Flatten those ordinary SPIR_FUNC helpers before
	// clspv performs opaque-pointer type inference and structured-CFG lowering;
	// otherwise helper boundaries lose the pointee information carried by the
	// original HIP C++ call site.
	for (Function & function : module)
	{
		if (function.isDeclaration() || !function.hasInternalLinkage() ||
			function.getCallingConv() == CallingConv::SPIR_KERNEL)
			continue;
		function.removeFnAttr(Attribute::InlineHint);
		function.removeFnAttr(Attribute::NoInline);
		function.removeFnAttr(Attribute::OptimizeNone);
		function.addFnAttr(Attribute::AlwaysInline);
	}
}

[[nodiscard]] bool normalizeKernel(Module & module, Function & kernel,
	KernelLayout & output_layout)
{
	LLVMContext & context = module.getContext();
	const DataLayout & data_layout = module.getDataLayout();
	const std::string entry = kernel.getName().str();
	const std::string body_name = entry + ".openfpm_body";

	if (!kernel.getReturnType()->isVoidTy())
	{
		errs() << "MoltenVK kernel must return void: " << entry << '\n';
		return false;
	}
	if (module.getFunction(body_name) != nullptr)
	{
		errs() << "MoltenVK kernel body name already exists: " << body_name << '\n';
		return false;
	}

	SmallVector<Type *, 16> argument_types;
	SmallVector<Align, 16> argument_alignments;
	argument_types.reserve(kernel.arg_size());
	argument_alignments.reserve(kernel.arg_size());
	for (Argument & argument : kernel.args())
	{
		Type * field_type = packedFieldType(argument, context);
		if (field_type == nullptr || !field_type->isSized())
		{
			errs() << "Unsized MoltenVK kernel argument " << argument.getArgNo()
				<< " in " << entry << '\n';
			return false;
		}
		Align alignment = data_layout.getABITypeAlign(field_type);
		if (argument.hasByValAttr())
		{
			alignment = Align(std::max(alignment.value(),
				argument.getParamAlign().valueOrOne().value()));
		}
		argument_types.push_back(field_type);
		argument_alignments.push_back(alignment);
	}

	// Compute a byte-addressed record explicitly instead of exposing a nested
	// aggregate as kernel POD.  This preserves HIP `byval(T) align N` contracts
	// while the SPIR kernel ABI remains one physical-storage-buffer pointer.
	std::uint64_t next_offset = 0;
	std::uint64_t block_alignment_value = 1;
	output_layout.entry = entry;
	output_layout.arguments.reserve(argument_types.size());
	for (std::size_t index = 0; index < argument_types.size(); ++index)
	{
		const std::uint64_t alignment = argument_alignments[index].value();
		block_alignment_value = std::max(block_alignment_value, alignment);
		const std::uint64_t field_offset = alignTo(next_offset, alignment);

		ArgumentLayout argument_layout;
		argument_layout.offset = field_offset;
		if (!fixedSize(data_layout.getTypeAllocSize(argument_types[index]),
			argument_layout.size, "kernel argument")) return false;
		argument_layout.alignment = alignment;
		output_layout.arguments.push_back(argument_layout);
		next_offset = field_offset + argument_layout.size;
	}
	output_layout.size = alignTo(next_offset, block_alignment_value);
	output_layout.alignment = block_alignment_value;

	const GlobalValue::LinkageTypes original_linkage = kernel.getLinkage();
	FunctionType * wrapper_type = FunctionType::get(Type::getVoidTy(context),
		{PointerType::get(context, 1)}, false);
	Function * wrapper = Function::Create(wrapper_type, original_linkage, "",
		&module);
	copyKernelProperties(*wrapper, kernel);
	wrapper->setCallingConv(CallingConv::SPIR_KERNEL);
	wrapper->addParamAttr(0, Attribute::getWithAlignment(
		context, Align(block_alignment_value)));
	wrapper->addFnAttr("openfpm-moltenvk-pointer-abi");

	// Preserve metadata/global references to the public entry before adding the
	// wrapper-to-body call.  Function pointers are opaque in the LLVM version
	// used by the HIP-to-SPIR-V toolchain, so the signatures remain RAUW-safe.
	kernel.replaceAllUsesWith(wrapper);
	kernel.setName(body_name);
	wrapper->setName(entry);

	if (wrapper->getName() != entry || kernel.getName() != body_name)
	{
		errs() << "Unable to preserve exact MoltenVK kernel entry name: " << entry
			<< '\n';
		return false;
	}

	kernel.setCallingConv(CallingConv::SPIR_FUNC);
	kernel.setLinkage(GlobalValue::InternalLinkage);
	kernel.setVisibility(GlobalValue::DefaultVisibility);
	kernel.setDLLStorageClass(GlobalValue::DefaultStorageClass);
	kernel.setComdat(nullptr);
	// The wrapper materializes original byval parameters in aligned private
	// storage.  Make its single body call mandatory-inline so clspv sees the
	// ordinary HIP kernel implementation behind the pointer-only entry ABI.
	kernel.removeFnAttr(Attribute::NoInline);
	kernel.removeFnAttr(Attribute::OptimizeNone);
	kernel.addFnAttr(Attribute::AlwaysInline);

	BasicBlock * entry_block = BasicBlock::Create(context, "entry", wrapper);
	IRBuilder<> builder(entry_block);
	Argument * packed_argument = wrapper->getArg(0);
	packed_argument->setName("openfpm_args");

	SmallVector<Value *, 16> call_arguments;
	call_arguments.reserve(kernel.arg_size());
	for (Argument & original_argument : kernel.args())
	{
		const unsigned index = original_argument.getArgNo();
		Type * field_type = argument_types[index];
		const std::uint64_t byte_offset =
			output_layout.arguments[index].offset;
		if (original_argument.hasByValAttr())
		{
			const Align alignment = argument_alignments[index];
			Value * field_value = loadPackedValue(field_type, packed_argument,
				byte_offset, Align(block_alignment_value), data_layout, builder);
			if (field_value == nullptr)
			{
				errs() << "Unsupported packed Metal byval argument " << index
					<< " in " << entry << '\n';
				return false;
			}
			AllocaInst * copy = builder.CreateAlloca(field_type, nullptr,
				"openfpm_byval_copy");
			copy->setAlignment(alignment);
			builder.CreateAlignedStore(field_value, copy, alignment);
			call_arguments.push_back(copy);
		}
		else if (original_argument.getType()->isPointerTy())
		{
			Value * address = loadPackedValue(field_type,packed_argument,
				byte_offset,Align(block_alignment_value),data_layout,builder);
			if (address == nullptr) return false;
			call_arguments.push_back(builder.CreateIntToPtr(address,
				original_argument.getType(), "openfpm_pointer"));
		}
		else
		{
			Value * value = loadPackedValue(field_type,packed_argument,
				byte_offset,Align(block_alignment_value),data_layout,builder);
			if (value == nullptr) return false;
			call_arguments.push_back(value);
		}
	}

	CallInst * call = builder.CreateCall(kernel.getFunctionType(), &kernel,
		call_arguments);
	call->setCallingConv(CallingConv::SPIR_FUNC);
	call->setAttributes(kernel.getAttributes());
	builder.CreateRetVoid();
	return true;
}

int normalizeLlvmCompatibility(StringRef input_path, StringRef output_path)
{
	LLVMContext context;
	SMDiagnostic diagnostic;
	std::unique_ptr<Module> module = parseIRFile(input_path, diagnostic, context);
	if (!module)
	{
		diagnostic.print("MoltenVKKernelABI", errs());
		return 1;
	}
	if (module->getDataLayout().isDefault())
	{
		errs() << "MoltenVKKernelABI: input module has no data layout\n";
		return 1;
	}

	// This mode runs after the mandatory inline/SROA pipeline.  At that point
	// pointer-containing byval aggregates have explicit leaf memory operations,
	// so their integer-address representation can be normalized losslessly.
	lowerUnsupportedDevicePrintf(*module);
	lowerPointerValuedPhysicalMemoryAccesses(*module);
	lowerPointerValuedMemoryPayloads(*module);
	foldInsertedAggregateExtractions(*module);
	lowerInlinedChipStarAtomicAdds(*module);
	lowerPhysicalMemoryAccesses(*module);
	lowerPhysicalPtrToAddr(*module);
	lowerPhysicalPointerComparisons(*module);
	foldInsertedAggregateExtractions(*module);
	lowerStorageOnlyDoubleTransfers(*module);
	lowerPromotedFloatArithmetic(*module);
	lowerPromotedFloatComparisons(*module);
	lowerAggregateGepPtrToInt(*module);
	lowerPhysicalPtrToInt(*module);
	lowerPackedArgumentFloatingLoads(*module);
	if (verifyModule(*module, &errs()))
	{
		errs() << "MoltenVKKernelABI: LLVM compatibility verification failed\n";
		return 1;
	}

	std::error_code error;
	raw_fd_ostream output(output_path, error, sys::fs::OF_Text);
	if (error)
	{
		errs() << "MoltenVKKernelABI: cannot write " << output_path << ": "
			<< error.message() << '\n';
		return 1;
	}
	module->print(output, nullptr);
	return 0;
}

int run(StringRef input_path, StringRef output_path, StringRef metadata_path)
{
	LLVMContext context;
	SMDiagnostic diagnostic;
	std::unique_ptr<Module> module = parseIRFile(input_path, diagnostic, context);
	if (!module)
	{
		diagnostic.print("MoltenVKKernelABI", errs());
		return 1;
	}
	if (module->getDataLayout().isDefault())
	{
		errs() << "MoltenVKKernelABI: input module has no data layout\n";
		return 1;
	}
	lowerChipStarAtomicAdds(*module);
	lowerPointerValuedPhysicalMemoryAccesses(*module);
	lowerPhysicalMemoryAccesses(*module);
	lowerPhysicalPtrToAddr(*module);
	lowerPhysicalPointerComparisons(*module);

	SmallVector<Function *, 16> kernels;
	for (Function & function : *module)
		if (!function.isDeclaration() &&
			function.getCallingConv() == CallingConv::SPIR_KERNEL)
			kernels.push_back(&function);
	if (kernels.empty())
	{
		errs() << "MoltenVKKernelABI: input contains no SPIR kernels\n";
		return 1;
	}

	std::vector<KernelLayout> layouts;
	layouts.reserve(kernels.size());
	for (Function * kernel : kernels)
	{
		KernelLayout layout;
		if (!normalizeKernel(*module, *kernel, layout)) return 1;
		layouts.push_back(std::move(layout));
	}
	markReachableDeviceFunctionsAlwaysInline(*module);

	if (verifyModule(*module, &errs()))
	{
		errs() << "MoltenVKKernelABI: transformed module verification failed\n";
		return 1;
	}

	std::error_code error;
	raw_fd_ostream output(output_path, error, sys::fs::OF_Text);
	if (error)
	{
		errs() << "MoltenVKKernelABI: cannot write " << output_path << ": "
			<< error.message() << '\n';
		return 1;
	}
	module->print(output, nullptr);
	output.close();

	raw_fd_ostream metadata(metadata_path, error, sys::fs::OF_Text);
	if (error)
	{
		errs() << "MoltenVKKernelABI: cannot write " << metadata_path << ": "
			<< error.message() << '\n';
		return 1;
	}
	for (const KernelLayout & layout : layouts) writeMetadata(metadata, layout);
	metadata.close();
	return 0;
}

} // namespace

int main(int argc, char ** argv)
{
	if (argc == 4 && StringRef(argv[1]) == "--llvm-compat")
		return normalizeLlvmCompatibility(argv[2], argv[3]);
	if (argc == 4 && StringRef(argv[1]) == "--spirv-compat")
		return normalizeMoltenVKSpirv(argv[2], argv[3]);
	if (argc == 4 && StringRef(argv[1]) == "--spirv-reflect")
		return writeSpirvPushConstantMetadata(argv[2], argv[3]);
	if (argc != 4)
	{
		llvm::errs() << "usage: " << argv[0]
			<< " input.ll output.ll metadata.jsonl\n"
			<< "       " << argv[0]
			<< " --llvm-compat input.ll output.ll\n"
			<< "       " << argv[0]
			<< " --spirv-compat input.spv output.spv\n";
		llvm::errs() << "       " << argv[0]
			<< " --spirv-reflect input.spv output.json\n";
		return 2;
	}
	return run(argv[1], argv[2], argv[3]);
}
