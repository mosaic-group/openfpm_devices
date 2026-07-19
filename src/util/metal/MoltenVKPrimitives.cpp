#include "MoltenVKPrimitives.hpp"

#include "memory/MoltenVKMemory.hpp"
#include "util/metal/MoltenVKContext.hpp"
#include "util/metal/MoltenVKKernel.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" void openfpm_register_moltenvk_primitives();

namespace openfpm
{
namespace metal
{
namespace
{
constexpr std::uint32_t threads_per_group = 256;

void ensure_primitives_registered()
{
	static const bool registered = []
	{
		openfpm_register_moltenvk_primitives();
		return true;
	}();
	(void)registered;
}

std::uint32_t checked_count(std::size_t count)
{
	if (count > std::numeric_limits<std::uint32_t>::max())
		throw std::overflow_error("MoltenVK primitive count exceeds uint32_t");
	return static_cast<std::uint32_t>(count);
}

std::uint32_t checked_product(std::size_t first, std::size_t second,
	const char * what)
{
	if (second != 0 && first > std::numeric_limits<std::size_t>::max() / second)
		throw std::overflow_error(what);
	return checked_count(first * second);
}

VkExtent3D groups_for(std::size_t count)
{
	return VkExtent3D{static_cast<std::uint32_t>((count + threads_per_group - 1) /
		threads_per_group),1,1};
}

template<typename Args>
void launch_1d(const char * name, std::size_t count, const Args & args)
{
	if (count == 0) return;
	launch_registered(name,groups_for(count),VkExtent3D{threads_per_group,1,1},args);
}

const char * type_suffix(primitive_type type)
{
	switch (type)
	{
	case primitive_type::i32: return "i32";
	case primitive_type::u32: return "u32";
	case primitive_type::u64: return "u64";
	case primitive_type::f32: return "f32";
	case primitive_type::i64: return "i64";
	case primitive_type::f64: break;
	}
	throw std::invalid_argument("unsupported MoltenVK primitive type");
}

template<typename T>
void exclusive_scan_t(const void * input, std::size_t count, void * output,
	primitive_type type)
{
	if (count == 0) return;
	MoltenVKMemory first(count * sizeof(T));
	MoltenVKMemory second(count * sizeof(T));
	const T * source = static_cast<const T *>(input);
	T * destination = static_cast<T *>(first.getDevicePointer());
	bool use_first = true;
	const std::string step_name = std::string("openfpm_scan_step_") + type_suffix(type);
	for (std::uint32_t offset = 1; offset < count; offset <<= 1)
	{
		primitive_abi::scan_args<T> args{source,destination,checked_count(count),offset};
		launch_1d(step_name.c_str(),count,args);
		source = destination;
		use_first = !use_first;
		destination = static_cast<T *>((use_first ? first : second).getDevicePointer());
		if (offset > std::numeric_limits<std::uint32_t>::max() / 2) break;
	}
	const std::string finish_name = std::string("openfpm_scan_exclusive_") + type_suffix(type);
	primitive_abi::scan_args<T> finish{source,static_cast<T *>(output),
		checked_count(count),0};
	launch_1d(finish_name.c_str(),count,finish);
}

template<typename T>
void reduce_t(const void * input, std::size_t count, void * output,
	primitive_type type, primitive_op operation, const void * initial_value)
{
	const T initial = *static_cast<const T *>(initial_value);
	MoltenVKMemory initial_storage(sizeof(T));
	std::memcpy(initial_storage.getPointer(),&initial,sizeof(T));
	initial_storage.hostToDevice();
	const T * initial_device = static_cast<const T *>(
		initial_storage.getDevicePointer());
	const std::string pair_name = std::string("openfpm_reduce_pair_") + type_suffix(type);
	const std::string finish_name = std::string("openfpm_reduce_finish_") + type_suffix(type);
	if (count == 0)
	{
		primitive_abi::reduce_args<T> finish{nullptr,static_cast<T *>(output),0,
			static_cast<std::uint32_t>(operation),initial_device};
		launch_registered(finish_name.c_str(),VkExtent3D{1,1,1},VkExtent3D{1,1,1},finish);
		return;
	}
	MoltenVKMemory first(count * sizeof(T));
	MoltenVKMemory second(count * sizeof(T));
	const T * source = static_cast<const T *>(input);
	T * destination = static_cast<T *>(first.getDevicePointer());
	bool use_first = true;
	std::size_t remaining = count;
	while (remaining > 1)
	{
		const std::size_t next = (remaining + 1) / 2;
		primitive_abi::reduce_args<T> args{source,destination,checked_count(remaining),
			static_cast<std::uint32_t>(operation),initial_device};
		launch_1d(pair_name.c_str(),next,args);
		source = destination;
		remaining = next;
		use_first = !use_first;
		destination = static_cast<T *>((use_first ? first : second).getDevicePointer());
	}
	primitive_abi::reduce_args<T> finish{source,static_cast<T *>(output),1,
		static_cast<std::uint32_t>(operation),initial_device};
	launch_registered(finish_name.c_str(),VkExtent3D{1,1,1},VkExtent3D{1,1,1},finish);
}

template<typename T>
void segmented_reduce_t(const void * input, std::size_t count,
	const void * segment_offsets, std::size_t segment_count, void * output,
	primitive_type type, primitive_op operation, const void * initial_value,
	std::size_t component_count)
{
	if (segment_count == 0) return;
	if (component_count == 0)
		throw std::invalid_argument("MoltenVK segmented reduction component count is zero");
	const T initial = *static_cast<const T *>(initial_value);
	MoltenVKMemory initial_storage(sizeof(T));
	std::memcpy(initial_storage.getPointer(),&initial,sizeof(T));
	initial_storage.hostToDevice();
	const std::string name = std::string("openfpm_segmented_reduce_") + type_suffix(type);
	primitive_abi::segmented_reduce_args<T> args{
		static_cast<const T *>(input),
		static_cast<const std::uint32_t *>(segment_offsets),
		static_cast<T *>(output),checked_count(count),checked_count(segment_count),
		checked_count(component_count),static_cast<std::uint32_t>(operation),
		static_cast<const T *>(initial_storage.getDevicePointer())};
	const std::uint32_t work_items = checked_product(segment_count,component_count,
		"MoltenVK segmented reduction work size overflow");
	launch_1d(name.c_str(),work_items,args);
}

void reduce_f64_shared(const void * input, std::size_t count, void * output,
	primitive_op operation, const void * initial_value)
{
	// Apple GPUs expose no native fp64 arithmetic. MoltenVK allocations are
	// persistently mapped shared buffers, so retain the public device-pointer
	// API while executing this compatibility case on the CPU after the queue is
	// quiescent. This is intentionally synchronous and correctness-first.
	metal_synchronize();
	if (count > std::numeric_limits<std::size_t>::max() / sizeof(double))
		throw std::overflow_error("MoltenVK fp64 reduction size overflow");
	const auto * source = count == 0 ? nullptr : static_cast<const double *>(
		host_pointer_from_device(input,count * sizeof(double)));
	auto * destination = static_cast<double *>(
		host_pointer_from_device(output,sizeof(double)));
	double value = *static_cast<const double *>(initial_value);
	for (std::size_t i = 0; i < count; ++i)
	{
		if (operation == primitive_op::minimum)
			value = source[i] < value ? source[i] : value;
		else if (operation == primitive_op::maximum)
			value = value < source[i] ? source[i] : value;
		else
			value += source[i];
	}
	*destination = value;
	flush_device_pointer(output,sizeof(double));
}

void segmented_reduce_f64_shared(const void * input, std::size_t count,
	const void * segment_offsets, std::size_t segment_count, void * output,
	primitive_op operation, const void * initial_value,
	std::size_t component_count)
{
	if (segment_count == 0) return;
	if (component_count == 0)
		throw std::invalid_argument("MoltenVK segmented reduction component count is zero");
	metal_synchronize();
	checked_count(count);
	checked_count(segment_count);
	checked_count(component_count);
	const std::size_t input_components = checked_product(count,component_count,
		"MoltenVK fp64 segmented reduction input size overflow");
	const std::size_t output_components = checked_product(segment_count,component_count,
		"MoltenVK fp64 segmented reduction output size overflow");
	if (input_components > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
		segment_count == std::numeric_limits<std::size_t>::max() ||
		segment_count + 1 > std::numeric_limits<std::size_t>::max() /
			sizeof(std::uint32_t) ||
		output_components > std::numeric_limits<std::size_t>::max() / sizeof(double))
		throw std::overflow_error("MoltenVK fp64 segmented reduction size overflow");
	const auto * source = input_components == 0 ? nullptr : static_cast<const double *>(
		host_pointer_from_device(input,static_cast<std::size_t>(input_components) * sizeof(double)));
	const auto * offsets = static_cast<const std::uint32_t *>(
		host_pointer_from_device(segment_offsets,
			(segment_count + 1) * sizeof(std::uint32_t)));
	auto * destination = static_cast<double *>(
		host_pointer_from_device(output,static_cast<std::size_t>(output_components) * sizeof(double)));
	const double initial = *static_cast<const double *>(initial_value);
	for (std::size_t segment = 0; segment < segment_count; ++segment)
	{
		const std::size_t begin = offsets[segment];
		const std::size_t end = offsets[segment+1] > count
			? count : static_cast<std::size_t>(offsets[segment+1]);
		for (std::size_t component = 0; component < component_count; ++component)
		{
			double value = initial;
			std::size_t i = begin;
			if (begin < end && (operation == primitive_op::left ||
				operation == primitive_op::right))
			{
				value = source[begin * component_count + component];
				i = begin + 1;
			}
			for (; i < end; ++i)
			{
				const double next = source[i * component_count + component];
				if (operation == primitive_op::minimum)
					value = next < value ? next : value;
				else if (operation == primitive_op::maximum)
					value = value < next ? next : value;
				else if (operation == primitive_op::right)
					value = next;
				else if (operation != primitive_op::left)
					value += next;
			}
			destination[segment * component_count + component] = value;
		}
	}
	flush_device_pointer(output,static_cast<std::size_t>(output_components) * sizeof(double));
}

template<typename Key, typename FlagArgs, typename ScatterArgs>
void stable_sort_pairs_t(void * keys, void * values, std::size_t count,
	primitive_type key_type, std::size_t value_size, bool descending,
	const char * flag_kernel, const char * scatter_kernel,
	std::uint32_t bit_count)
{
	MoltenVKMemory temporary_keys(count * sizeof(Key));
	MoltenVKMemory temporary_values(count * value_size);
	MoltenVKMemory flags(count * sizeof(std::uint32_t));
	MoltenVKMemory prefix(count * sizeof(std::uint32_t));
	auto * source_keys = static_cast<Key *>(keys);
	auto * source_values = static_cast<std::uint8_t *>(values);
	auto * destination_keys = static_cast<Key *>(temporary_keys.getDevicePointer());
	auto * destination_values = static_cast<std::uint8_t *>(temporary_values.getDevicePointer());
	for (std::uint32_t bit = 0; bit != bit_count; ++bit)
	{
		FlagArgs flag_args{source_keys,
			static_cast<std::uint32_t *>(flags.getDevicePointer()),checked_count(count),bit,
			static_cast<std::uint32_t>(key_type),descending ? 1u : 0u};
		launch_1d(flag_kernel,count,flag_args);
		exclusive_scan(flags.getDevicePointer(),count,prefix.getDevicePointer(),
			primitive_type::u32);
		ScatterArgs scatter_args{source_keys,source_values,
			destination_keys,destination_values,
			static_cast<const std::uint32_t *>(flags.getDevicePointer()),
			static_cast<const std::uint32_t *>(prefix.getDevicePointer()),
			checked_count(count),checked_count(value_size)};
		launch_1d(scatter_kernel,count,scatter_args);
		std::swap(source_keys,destination_keys);
		std::swap(source_values,destination_values);
	}
	// Both supported key widths have an even radix pass count, so the final
	// stable ordering is back in the caller-owned buffers.
	if (source_keys != keys)
		throw std::logic_error("Metal radix sort did not finish in the input buffers");
}

}

void exclusive_scan(const void * input, std::size_t count, void * output,
	primitive_type type)
{
	if (type == primitive_type::f64)
		throw std::invalid_argument("Metal fp64 scan is not supported");
	ensure_primitives_registered();
	switch (type)
	{
	case primitive_type::i32: exclusive_scan_t<std::int32_t>(input,count,output,type); break;
	case primitive_type::u32: exclusive_scan_t<std::uint32_t>(input,count,output,type); break;
	case primitive_type::u64: exclusive_scan_t<std::uint64_t>(input,count,output,type); break;
	case primitive_type::f32: exclusive_scan_t<float>(input,count,output,type); break;
	case primitive_type::i64: exclusive_scan_t<std::int64_t>(input,count,output,type); break;
	case primitive_type::f64: break;
	}
}

void reduce(const void * input, std::size_t count, void * output,
	primitive_type type, primitive_op operation, const void * initial_value)
{
	if (type == primitive_type::f64)
	{
		reduce_f64_shared(input,count,output,operation,initial_value);
		return;
	}
	ensure_primitives_registered();
	switch (type)
	{
	case primitive_type::i32: reduce_t<std::int32_t>(input,count,output,type,operation,initial_value); break;
	case primitive_type::u32: reduce_t<std::uint32_t>(input,count,output,type,operation,initial_value); break;
	case primitive_type::u64: reduce_t<std::uint64_t>(input,count,output,type,operation,initial_value); break;
	case primitive_type::f32: reduce_t<float>(input,count,output,type,operation,initial_value); break;
	case primitive_type::i64: reduce_t<std::int64_t>(input,count,output,type,operation,initial_value); break;
	case primitive_type::f64: break;
	}
}

void segmented_reduce(const void * input, std::size_t count,
	const void * segment_offsets, std::size_t segment_count, void * output,
	primitive_type type, primitive_op operation, const void * initial_value,
	std::size_t component_count)
{
	if (type == primitive_type::f64)
	{
		segmented_reduce_f64_shared(input,count,segment_offsets,segment_count,
			output,operation,initial_value,component_count);
		return;
	}
	ensure_primitives_registered();
	switch (type)
	{
	case primitive_type::i32: segmented_reduce_t<std::int32_t>(input,count,segment_offsets,segment_count,output,type,operation,initial_value,component_count); break;
	case primitive_type::u32: segmented_reduce_t<std::uint32_t>(input,count,segment_offsets,segment_count,output,type,operation,initial_value,component_count); break;
	case primitive_type::u64: segmented_reduce_t<std::uint64_t>(input,count,segment_offsets,segment_count,output,type,operation,initial_value,component_count); break;
	case primitive_type::f32: segmented_reduce_t<float>(input,count,segment_offsets,segment_count,output,type,operation,initial_value,component_count); break;
	case primitive_type::i64: segmented_reduce_t<std::int64_t>(input,count,segment_offsets,segment_count,output,type,operation,initial_value,component_count); break;
	case primitive_type::f64: break;
	}
}

void stable_sort_pairs(void * keys, void * values, std::size_t count,
	primitive_type key_type, std::size_t value_size, bool descending)
{
	if (key_type == primitive_type::f64)
		throw std::invalid_argument("Metal fp64 sort is not supported");
	ensure_primitives_registered();
	if (count < 2) return;
	checked_count(count);
	if (value_size == 0) throw std::invalid_argument("sort value size is zero");
	if (key_type == primitive_type::u64 || key_type == primitive_type::i64)
		stable_sort_pairs_t<std::uint64_t,primitive_abi::radix_flag_args_u64,
			primitive_abi::radix_scatter_args_u64>(keys,values,count,key_type,
			value_size,descending,"openfpm_radix_flags_u64",
			"openfpm_radix_scatter_u64",64);
	else
		stable_sort_pairs_t<std::uint32_t,primitive_abi::radix_flag_args,
			primitive_abi::radix_scatter_args>(keys,values,count,key_type,
			value_size,descending,"openfpm_radix_flags_u32",
			"openfpm_radix_scatter_u32",32);
}

void merge_pairs(const void * a_keys, const void * a_values, std::size_t a_count,
	const void * b_keys, const void * b_values, std::size_t b_count,
	void * output_keys, void * output_values, primitive_type key_type,
	std::size_t value_size, bool descending)
{
	if (key_type == primitive_type::f64)
		throw std::invalid_argument("Metal fp64 merge is not supported");
	ensure_primitives_registered();
	const std::size_t total = a_count + b_count;
	if (total == 0) return;
	if (key_type == primitive_type::u64 || key_type == primitive_type::i64)
	{
		primitive_abi::merge_args_u64 args{
			static_cast<const std::uint64_t *>(a_keys),static_cast<const std::uint8_t *>(a_values),
			static_cast<const std::uint64_t *>(b_keys),static_cast<const std::uint8_t *>(b_values),
			static_cast<std::uint64_t *>(output_keys),static_cast<std::uint8_t *>(output_values),
			checked_count(a_count),checked_count(b_count),static_cast<std::uint32_t>(key_type),
			checked_count(value_size),descending ? 1u : 0u};
		launch_1d("openfpm_merge_pairs_u64",total,args);
	}
	else
	{
		primitive_abi::merge_args args{
			static_cast<const std::uint32_t *>(a_keys),static_cast<const std::uint8_t *>(a_values),
			static_cast<const std::uint32_t *>(b_keys),static_cast<const std::uint8_t *>(b_values),
			static_cast<std::uint32_t *>(output_keys),static_cast<std::uint8_t *>(output_values),
			checked_count(a_count),checked_count(b_count),static_cast<std::uint32_t>(key_type),
			checked_count(value_size),descending ? 1u : 0u};
		launch_1d("openfpm_merge_pairs_u32",total,args);
	}
}

} // namespace metal
} // namespace openfpm
