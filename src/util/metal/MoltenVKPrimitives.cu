#include "util/metal/MoltenVKPrimitives.hpp"

namespace openfpm
{
namespace metal
{
namespace primitive_device
{

template<typename T>
__device__ typename std::enable_if<std::is_integral<T>::value,T>::type
bitwise_or(T a, T b)
{
	return a | b;
}

template<typename T>
__device__ typename std::enable_if<!std::is_integral<T>::value,T>::type
bitwise_or(T a, T)
{
	// The host dispatch rejects bitwise operations for non-integral storage.
	// Keeping this overload well-formed lets the shared typed kernel compile.
	return a;
}

template<typename T>
__device__ T apply(T a, T b, std::uint32_t operation)
{
	if (operation == static_cast<std::uint32_t>(primitive_op::minimum))
		return b < a ? b : a;
	if (operation == static_cast<std::uint32_t>(primitive_op::maximum))
		return a < b ? b : a;
	if (operation == static_cast<std::uint32_t>(primitive_op::right))
		return b;
	if (operation == static_cast<std::uint32_t>(primitive_op::left))
		return a;
	if (operation == static_cast<std::uint32_t>(primitive_op::bitwise_or))
		return bitwise_or(a,b);
	return a + b;
}

template<typename T>
__device__ void scan_step(primitive_abi::scan_args<T> args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= args.count) return;
	const auto input = args.input;
	auto output = args.output;
	T value = input[id];
	if (id >= args.offset) value = value + input[id - args.offset];
	output[id] = value;
}

template<typename T>
__device__ void scan_exclusive(primitive_abi::scan_args<T> args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= args.count) return;
	const auto input = args.input;
	auto output = args.output;
	output[id] = id == 0 ? T(0) : input[id - 1];
}

template<typename T>
__device__ void reduce_pair(primitive_abi::reduce_args<T> args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t first = id * 2;
	if (first >= args.count) return;
	const auto input = args.input;
	auto output = args.output;
	T value = input[first];
	if (first + 1 < args.count) value = apply(value,input[first + 1],args.operation);
	output[id] = value;
}

template<typename T>
__device__ void reduce_finish(primitive_abi::reduce_args<T> args)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	auto output = args.output;
	if (args.count == 0)
	{
		output[0] = *args.initial;
		return;
	}
	const auto input = args.input;
	output[0] = apply(*args.initial,input[0],args.operation);
}

template<typename T>
__device__ void segmented(primitive_abi::segmented_reduce_args<T> args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t total = args.segment_count * args.component_count;
	if (id >= total) return;
	const std::uint32_t segment = id / args.component_count;
	const std::uint32_t component = id - segment * args.component_count;
	const auto input = args.input;
	const auto offsets = args.segments;
	auto output = args.output;
	std::uint32_t begin = offsets[segment];
	std::uint32_t end = offsets[segment + 1];
	if (begin > args.count) begin = args.count;
	if (end > args.count) end = args.count;
	T value = *args.initial;
	if (begin < end &&
		(args.operation == static_cast<std::uint32_t>(primitive_op::left) ||
		 args.operation == static_cast<std::uint32_t>(primitive_op::right)))
	{
		value = input[begin * args.component_count + component];
		++begin;
	}
	for (std::uint32_t i = begin; i < end; ++i)
		value = apply(value,input[i * args.component_count + component],args.operation);
	output[segment * args.component_count + component] = value;
}

__device__ std::uint32_t ordered_key(std::uint32_t bits,
	std::uint32_t key_type, std::uint32_t descending)
{
	std::uint32_t ordered = bits;
	if (key_type == static_cast<std::uint32_t>(primitive_type::i32))
		ordered ^= 0x80000000u;
	else if (key_type == static_cast<std::uint32_t>(primitive_type::f32))
		ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
	return descending ? ~ordered : ordered;
}

__device__ std::uint64_t ordered_key(std::uint64_t bits,
	std::uint32_t key_type, std::uint32_t descending)
{
	std::uint64_t ordered = bits;
	if (key_type == static_cast<std::uint32_t>(primitive_type::i64))
		ordered ^= 0x8000000000000000ull;
	return descending ? ~ordered : ordered;
}

__device__ void copy_value(
	std::uint8_t * destination, const std::uint8_t * source,
	std::uint32_t destination_index, std::uint32_t source_index,
	std::uint32_t value_size)
{
	const std::uint32_t destination_offset = destination_index * value_size;
	const std::uint32_t source_offset = source_index * value_size;
	for (std::uint32_t byte = 0; byte < value_size; ++byte)
		destination[destination_offset + byte] = source[source_offset + byte];
}

} // namespace primitive_device

#define OPENFPM_DEFINE_TYPED_PRIMITIVES(suffix, type) \
__global__ void openfpm_scan_step_##suffix(primitive_abi::scan_args<type> args) \
{ primitive_device::scan_step(args); } \
__global__ void openfpm_scan_exclusive_##suffix(primitive_abi::scan_args<type> args) \
{ primitive_device::scan_exclusive(args); } \
__global__ void openfpm_reduce_pair_##suffix(primitive_abi::reduce_args<type> args) \
{ primitive_device::reduce_pair(args); } \
__global__ void openfpm_reduce_finish_##suffix(primitive_abi::reduce_args<type> args) \
{ primitive_device::reduce_finish(args); } \
__global__ void openfpm_segmented_reduce_##suffix(primitive_abi::segmented_reduce_args<type> args) \
{ primitive_device::segmented(args); }

OPENFPM_DEFINE_TYPED_PRIMITIVES(i32, std::int32_t)
OPENFPM_DEFINE_TYPED_PRIMITIVES(u32, std::uint32_t)
OPENFPM_DEFINE_TYPED_PRIMITIVES(u64, std::uint64_t)
OPENFPM_DEFINE_TYPED_PRIMITIVES(f32, float)
OPENFPM_DEFINE_TYPED_PRIMITIVES(i64, std::int64_t)

__global__ void openfpm_radix_flags_u32(primitive_abi::radix_flag_args args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= args.count) return;
	const auto keys = args.keys;
	auto flags = args.flags;
	const std::uint32_t ordered = primitive_device::ordered_key(
		keys[id],args.key_type,args.descending);
	flags[id] = ((ordered >> args.bit) & 1u) == 0u ? 1u : 0u;
}

__global__ void openfpm_radix_scatter_u32(primitive_abi::radix_scatter_args args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= args.count) return;
	const auto input_keys = args.input_keys;
	const auto input_values = args.input_values;
	auto output_keys = args.output_keys;
	auto output_values = args.output_values;
	const auto flags = args.flags;
	const auto prefix = args.prefix;
	const std::uint32_t zero_count = prefix[args.count - 1] + flags[args.count - 1];
	const std::uint32_t destination = flags[id] != 0
		? prefix[id] : zero_count + id - prefix[id];
	output_keys[destination] = input_keys[id];
	primitive_device::copy_value(output_values,input_values,destination,id,args.value_size);
}

__global__ void openfpm_radix_flags_u64(primitive_abi::radix_flag_args_u64 args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= args.count) return;
	const auto keys = args.keys;
	auto flags = args.flags;
	const std::uint64_t ordered = primitive_device::ordered_key(
		keys[id],args.key_type,args.descending);
	flags[id] = ((ordered >> args.bit) & 1ull) == 0ull ? 1u : 0u;
}

__global__ void openfpm_radix_scatter_u64(primitive_abi::radix_scatter_args_u64 args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= args.count) return;
	const auto input_keys = args.input_keys;
	const auto input_values = args.input_values;
	auto output_keys = args.output_keys;
	auto output_values = args.output_values;
	const auto flags = args.flags;
	const auto prefix = args.prefix;
	const std::uint32_t zero_count = prefix[args.count - 1] + flags[args.count - 1];
	const std::uint32_t destination = flags[id] != 0
		? prefix[id] : zero_count + id - prefix[id];
	output_keys[destination] = input_keys[id];
	primitive_device::copy_value(output_values,input_values,destination,id,args.value_size);
}

__global__ void openfpm_merge_pairs_u32(primitive_abi::merge_args args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t total = args.a_count + args.b_count;
	if (id >= total) return;
	const auto a_keys = args.a_keys;
	const auto b_keys = args.b_keys;
	const auto a_values = args.a_values;
	const auto b_values = args.b_values;
	auto output_keys = args.output_keys;
	auto output_values = args.output_values;

	const bool from_a = id < args.a_count;
	const std::uint32_t source_index = from_a ? id : id - args.a_count;
	const std::uint32_t key = from_a ? a_keys[source_index] : b_keys[source_index];
	const std::uint32_t ordered = primitive_device::ordered_key(key,args.key_type,args.descending);
	const auto other_keys = from_a ? b_keys : a_keys;
	const std::uint32_t other_count = from_a ? args.b_count : args.a_count;
	std::uint32_t low = 0, high = other_count;
	while (low < high)
	{
		const std::uint32_t middle = low + (high - low) / 2;
		const std::uint32_t other = primitive_device::ordered_key(
			other_keys[middle],args.key_type,args.descending);
		// A precedes B for equivalent keys, preserving merge_by_key stability.
		const bool advance = from_a ? other < ordered : other <= ordered;
		if (advance) low = middle + 1; else high = middle;
	}
	const std::uint32_t destination = source_index + low;
	output_keys[destination] = key;
	primitive_device::copy_value(output_values,from_a ? a_values : b_values,
		destination,source_index,args.value_size);
}

__global__ void openfpm_merge_pairs_u64(primitive_abi::merge_args_u64 args)
{
	const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t total = args.a_count + args.b_count;
	if (id >= total) return;
	const auto a_keys = args.a_keys;
	const auto b_keys = args.b_keys;
	const auto a_values = args.a_values;
	const auto b_values = args.b_values;
	auto output_keys = args.output_keys;
	auto output_values = args.output_values;

	const bool from_a = id < args.a_count;
	const std::uint32_t source_index = from_a ? id : id - args.a_count;
	const std::uint64_t key = from_a ? a_keys[source_index] : b_keys[source_index];
	const std::uint64_t ordered = primitive_device::ordered_key(key,args.key_type,args.descending);
	const auto other_keys = from_a ? b_keys : a_keys;
	const std::uint32_t other_count = from_a ? args.b_count : args.a_count;
	std::uint32_t low = 0, high = other_count;
	while (low < high)
	{
		const std::uint32_t middle = low + (high - low) / 2;
		const std::uint64_t other = primitive_device::ordered_key(
			other_keys[middle],args.key_type,args.descending);
		const bool advance = from_a ? other < ordered : other <= ordered;
		if (advance) low = middle + 1; else high = middle;
	}
	const std::uint32_t destination = source_index + low;
	output_keys[destination] = key;
	primitive_device::copy_value(output_values,from_a ? a_values : b_values,
		destination,source_index,args.value_size);
}

} // namespace metal
} // namespace openfpm
