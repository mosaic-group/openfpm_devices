#ifndef OPENFPM_MOLTENVK_PRIMITIVES_HPP_
#define OPENFPM_MOLTENVK_PRIMITIVES_HPP_

#include "util/cuda_util.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

// These are the reduction functors used by openfpm_data.  The Metal backend
// only needs their types here; their definitions remain in the existing
// CUDA/HIP data headers.
template<typename T> struct rightOperand_t;
template<typename T> struct leftOperand_t;
template<typename T> struct bitwiseOr_t;
template<typename T, unsigned int N> struct plus_block_t;
template<typename T, unsigned int N> struct minimum_block_t;
template<typename T, unsigned int N> struct maximum_block_t;

namespace openfpm
{
namespace metal
{

enum class primitive_type : std::uint32_t
{
	i32,
	u32,
	u64,
	f32,
	i64,
	f64
};

enum class primitive_op : std::uint32_t
{
	plus,
	minimum,
	maximum,
	right,
	left,
	bitwise_or
};

template<typename T> struct primitive_type_of;
template<> struct primitive_type_of<int> { static constexpr primitive_type value = primitive_type::i32; };
template<> struct primitive_type_of<unsigned int> { static constexpr primitive_type value = primitive_type::u32; };
template<> struct primitive_type_of<long> { static constexpr primitive_type value = primitive_type::i64; };
template<> struct primitive_type_of<long long> { static constexpr primitive_type value = primitive_type::i64; };
template<> struct primitive_type_of<unsigned long> { static constexpr primitive_type value = primitive_type::u64; };
template<> struct primitive_type_of<unsigned long long> { static constexpr primitive_type value = primitive_type::u64; };
template<> struct primitive_type_of<float> { static constexpr primitive_type value = primitive_type::f32; };
template<> struct primitive_type_of<double> { static constexpr primitive_type value = primitive_type::f64; };

template<typename T, typename = void>
struct is_supported_primitive : std::false_type {};

template<typename T>
struct is_supported_primitive<T, std::void_t<decltype(
	primitive_type_of<typename std::remove_cv<T>::type>::value)>> : std::true_type {};

// Metal GPUs do not provide native fp64. Double is accepted only by reduction
// compatibility paths, which synchronize and reduce through the shared
// MoltenVK allocation mapping on the CPU.
template<typename T>
struct is_supported_device_primitive : std::integral_constant<bool,
	is_supported_primitive<T>::value &&
	!std::is_same<typename std::remove_cv<T>::type,double>::value> {};

// Describe the scalar storage used by one logical segmented-reduction value.
// Scalar values have one component. C arrays and OpenFPM block values expose
// all of their tightly packed scalar components to the same primitive kernel;
// this keeps the public HIP/CUDA iterator signature unchanged.
template<typename T, typename = void>
struct segmented_value_traits
{
	using scalar_type = typename std::remove_cv<T>::type;
	static constexpr std::size_t component_count = 1;
	static constexpr bool tightly_packed = true;
};

template<typename T>
struct segmented_value_traits<T,std::void_t<
	typename std::remove_cv<T>::type::scalarType,
	decltype(std::remove_cv<T>::type::size)>>
{
	using value_type = typename std::remove_cv<T>::type;
	using scalar_type = typename std::remove_cv<
		typename value_type::scalarType>::type;
	static constexpr std::size_t component_count = value_type::size;
	static constexpr bool tightly_packed =
		sizeof(value_type) == component_count * sizeof(scalar_type);
};

template<typename T, std::size_t N>
struct segmented_value_traits<T[N],void>
{
	using nested_traits = segmented_value_traits<T>;
	using scalar_type = typename nested_traits::scalar_type;
	static constexpr std::size_t component_count =
		N * nested_traits::component_count;
	static constexpr bool tightly_packed = nested_traits::tightly_packed &&
		sizeof(T[N]) == component_count * sizeof(scalar_type);
};

template<typename T>
struct is_supported_segmented_value : std::integral_constant<bool,
	segmented_value_traits<typename std::remove_cv<T>::type>::tightly_packed &&
	is_supported_primitive<typename segmented_value_traits<
		typename std::remove_cv<T>::type>::scalar_type>::value> {};

// Operation traits are intentionally open for specialisation by the core data
// headers. This lets the existing right/left/bitwise and block functors retain
// their CUDA/HIP types while the Metal primitive layer selects an equivalent
// backend operation.
template<typename Op, typename Value>
struct segmented_operation_traits
{
	static constexpr bool supported = false;
	static constexpr primitive_op operation = primitive_op::plus;
	static constexpr std::size_t required_components = 0;
};

template<typename T>
struct segmented_operation_traits<gpu::plus_t<T>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::plus;
	static constexpr std::size_t required_components = 0;
};

template<typename T>
struct segmented_operation_traits<gpu::minimum_t<T>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::minimum;
	static constexpr std::size_t required_components = 0;
};

template<typename T>
struct segmented_operation_traits<gpu::maximum_t<T>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::maximum;
	static constexpr std::size_t required_components = 0;
};

template<typename T>
struct segmented_operation_traits<::rightOperand_t<T>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::right;
	static constexpr std::size_t required_components = 0;
};

template<typename T>
struct segmented_operation_traits<::leftOperand_t<T>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::left;
	static constexpr std::size_t required_components = 0;
};

template<typename T>
struct segmented_operation_traits<::bitwiseOr_t<T>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::bitwise_or;
	static constexpr std::size_t required_components = 0;
};

template<typename T, unsigned int N>
struct segmented_operation_traits<::plus_block_t<T,N>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::plus;
	static constexpr std::size_t required_components = N;
};

template<typename T, unsigned int N>
struct segmented_operation_traits<::minimum_block_t<T,N>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::minimum;
	static constexpr std::size_t required_components = N;
};

template<typename T, unsigned int N>
struct segmented_operation_traits<::maximum_block_t<T,N>,T>
{
	static constexpr bool supported = true;
	static constexpr primitive_op operation = primitive_op::maximum;
	static constexpr std::size_t required_components = N;
};

void exclusive_scan(const void * input, std::size_t count, void * output,
	primitive_type type);
void reduce(const void * input, std::size_t count, void * output,
	primitive_type type, primitive_op operation, const void * initial_value);
void segmented_reduce(const void * input, std::size_t count,
	const void * segment_offsets, std::size_t segment_count, void * output,
	primitive_type type, primitive_op operation, const void * initial_value,
	std::size_t component_count = 1);
void stable_sort_pairs(void * keys, void * values, std::size_t count,
	primitive_type key_type, std::size_t value_size, bool descending);
void merge_pairs(const void * a_keys, const void * a_values, std::size_t a_count,
	const void * b_keys, const void * b_values, std::size_t b_count,
	void * output_keys, void * output_values, primitive_type key_type,
	std::size_t value_size, bool descending);

// Declared by MoltenVKMemory.  The facade uses the coherent mapping only to
// read CUB's final segmented-reduction offset on the host dispatch side.
void * host_pointer_from_device(const void * device_pointer, std::size_t bytes);

namespace primitive_abi
{
template<typename T>
struct scan_args
{
	const T * input;
	T * output;
	std::uint32_t count;
	std::uint32_t offset;
};

template<typename T>
struct reduce_args
{
	const T * input;
	T * output;
	std::uint32_t count;
	std::uint32_t operation;
	const T * initial;
};

template<typename T>
struct segmented_reduce_args
{
	const T * input;
	const std::uint32_t * segments;
	T * output;
	std::uint32_t count;
	std::uint32_t segment_count;
	std::uint32_t component_count;
	std::uint32_t operation;
	const T * initial;
};

struct radix_flag_args
{
	const std::uint32_t * keys;
	std::uint32_t * flags;
	std::uint32_t count;
	std::uint32_t bit;
	std::uint32_t key_type;
	std::uint32_t descending;
};

struct radix_scatter_args
{
	const std::uint32_t * input_keys;
	const std::uint8_t * input_values;
	std::uint32_t * output_keys;
	std::uint8_t * output_values;
	const std::uint32_t * flags;
	const std::uint32_t * prefix;
	std::uint32_t count;
	std::uint32_t value_size;
};

struct radix_flag_args_u64
{
	const std::uint64_t * keys;
	std::uint32_t * flags;
	std::uint32_t count;
	std::uint32_t bit;
	std::uint32_t key_type;
	std::uint32_t descending;
};

struct radix_scatter_args_u64
{
	const std::uint64_t * input_keys;
	const std::uint8_t * input_values;
	std::uint64_t * output_keys;
	std::uint8_t * output_values;
	const std::uint32_t * flags;
	const std::uint32_t * prefix;
	std::uint32_t count;
	std::uint32_t value_size;
};

struct merge_args
{
	const std::uint32_t * a_keys;
	const std::uint8_t * a_values;
	const std::uint32_t * b_keys;
	const std::uint8_t * b_values;
	std::uint32_t * output_keys;
	std::uint8_t * output_values;
	std::uint32_t a_count;
	std::uint32_t b_count;
	std::uint32_t key_type;
	std::uint32_t value_size;
	std::uint32_t descending;
};

struct merge_args_u64
{
	const std::uint64_t * a_keys;
	const std::uint8_t * a_values;
	const std::uint64_t * b_keys;
	const std::uint8_t * b_values;
	std::uint64_t * output_keys;
	std::uint8_t * output_values;
	std::uint32_t a_count;
	std::uint32_t b_count;
	std::uint32_t key_type;
	std::uint32_t value_size;
	std::uint32_t descending;
};
}

} // namespace metal
} // namespace openfpm

// A deliberately narrow hipCUB facade for the OpenFPM primitive calls.  It
// preserves hipCUB's query-then-execute contract, allowing the shared core
// headers to use their normal HIP path.  It does not claim general hipCUB
// compatibility: only the operations and raw-pointer iterator forms used by
// OpenFPM are provided.
namespace openfpm_metal_hipcub_detail
{
template<typename Iterator>
struct iterator_value
{
	static_assert(std::is_pointer<Iterator>::value,
		"The Metal hipCUB facade supports raw device-pointer iterators only");
	using type = typename std::remove_cv<typename std::remove_pointer<Iterator>::type>::type;
};

template<typename InputIterator, typename OutputIterator>
struct matching_iterator_value
{
	using input_type = typename iterator_value<InputIterator>::type;
	using output_type = typename iterator_value<OutputIterator>::type;
	static_assert(std::is_same<input_type,output_type>::value,
		"Metal primitive input and output value types must match");
	using type = input_type;
};

inline std::size_t count_or_zero(int count)
{
	return count <= 0 ? 0 : static_cast<std::size_t>(count);
}

template<typename T, typename Operation>
struct scalar_operation
{
	static constexpr bool is_plus = std::is_base_of<gpu::plus_t<T>,Operation>::value;
	static constexpr bool is_minimum = std::is_base_of<gpu::minimum_t<T>,Operation>::value;
	static constexpr bool is_maximum = std::is_base_of<gpu::maximum_t<T>,Operation>::value;
	static_assert(is_plus || is_minimum || is_maximum,
		"The Metal hipCUB facade supports plus, minimum, and maximum reductions");
	static constexpr openfpm::metal::primitive_op value = is_minimum
		? openfpm::metal::primitive_op::minimum
		: (is_maximum ? openfpm::metal::primitive_op::maximum
			: openfpm::metal::primitive_op::plus);
};

template<typename Scalar, typename Initial>
typename std::enable_if<std::is_convertible<Initial,Scalar>::value,Scalar>::type
initial_scalar(const Initial & value)
{
	return static_cast<Scalar>(value);
}

template<typename Scalar, typename Initial>
typename std::enable_if<!std::is_convertible<Initial,Scalar>::value,Scalar>::type
initial_scalar(const Initial & value)
{
	static_assert(std::is_standard_layout<Initial>::value,
		"Metal block reduction initial value must use standard-layout scalar storage");
	return *reinterpret_cast<const Scalar *>(&value);
}

template<typename Comparator, typename Key>
struct comparator
{
	static constexpr bool ascending = std::is_base_of<gpu::less_t<Key>,Comparator>::value;
	static constexpr bool descending = std::is_base_of<gpu::greater_t<Key>,Comparator>::value;
	static_assert(ascending || descending,
		"The Metal primitive facade requires gpu::less_t or gpu::greater_t");
};

inline void query_storage(std::size_t & bytes)
{
	// The implementation owns its real scratch allocations.  A nonzero byte
	// preserves CUB's contract and gives ofp_context a stable execution token.
	bytes = 1;
}
}

namespace hipcub
{
struct DeviceScan
{
	template<typename InputIterator, typename OutputIterator>
	static int ExclusiveSum(void * temporary_storage, std::size_t & temporary_storage_bytes,
		InputIterator input, OutputIterator output, int count)
	{
		using value_type = typename openfpm_metal_hipcub_detail::matching_iterator_value<
			InputIterator,OutputIterator>::type;
		static_assert(openfpm::metal::is_supported_device_primitive<value_type>::value,
			"Metal exclusive scan supports 32/64-bit integers and float");
		if (temporary_storage == nullptr)
		{
			openfpm_metal_hipcub_detail::query_storage(temporary_storage_bytes);
			return 0;
		}
		openfpm::metal::exclusive_scan(input,
			openfpm_metal_hipcub_detail::count_or_zero(count),output,
			openfpm::metal::primitive_type_of<value_type>::value);
		return 0;
	}
};

struct DeviceReduce
{
	template<typename InputIterator, typename OutputIterator, typename Operation,
		typename Initial>
	static int Reduce(void * temporary_storage, std::size_t & temporary_storage_bytes,
		InputIterator input, OutputIterator output, int count, Operation,
		Initial initial)
	{
		using value_type = typename openfpm_metal_hipcub_detail::matching_iterator_value<
			InputIterator,OutputIterator>::type;
		static_assert(openfpm::metal::is_supported_primitive<value_type>::value,
			"Metal reduction supports 32/64-bit integers, float, and shared-memory double");
		if (temporary_storage == nullptr)
		{
			openfpm_metal_hipcub_detail::query_storage(temporary_storage_bytes);
			return 0;
		}
		const value_type initial_value = static_cast<value_type>(initial);
		openfpm::metal::reduce(input,
			openfpm_metal_hipcub_detail::count_or_zero(count),output,
			openfpm::metal::primitive_type_of<value_type>::value,
			openfpm_metal_hipcub_detail::scalar_operation<value_type,Operation>::value,
			&initial_value);
		return 0;
	}
};

struct DeviceSegmentedReduce
{
	template<typename InputIterator, typename OutputIterator,
		typename SegmentIterator, typename Operation, typename Initial>
	static int Reduce(void * temporary_storage, std::size_t & temporary_storage_bytes,
		InputIterator input, OutputIterator output, int segment_count,
		SegmentIterator begin_offsets, SegmentIterator, Operation, Initial initial)
	{
		using input_type = typename openfpm_metal_hipcub_detail::iterator_value<InputIterator>::type;
		using output_type = typename openfpm_metal_hipcub_detail::iterator_value<OutputIterator>::type;
		using segment_type = typename openfpm_metal_hipcub_detail::iterator_value<SegmentIterator>::type;
		using value_traits = openfpm::metal::segmented_value_traits<input_type>;
		using scalar_type = typename value_traits::scalar_type;
		using operation_traits = openfpm::metal::segmented_operation_traits<Operation,input_type>;
		static_assert(std::is_same<input_type,output_type>::value,
			"Metal segmented reduction value types must match");
		static_assert(sizeof(segment_type) == sizeof(std::uint32_t) &&
			std::is_integral<segment_type>::value,
			"Metal segmented reduction requires 32-bit integral offsets");
		static_assert(openfpm::metal::is_supported_segmented_value<input_type>::value,
			"Metal segmented reduction requires tightly packed supported scalar components");
		static_assert(operation_traits::supported,
			"The Metal hipCUB facade does not recognize this segmented reduction functor");
		static_assert(operation_traits::required_components == 0 ||
			operation_traits::required_components == value_traits::component_count,
			"Metal block reduction length does not match the value component count");
		static_assert(operation_traits::operation != openfpm::metal::primitive_op::bitwise_or ||
			std::is_integral<scalar_type>::value,
			"Metal bitwise segmented reduction requires integral components");
		if (temporary_storage == nullptr)
		{
			openfpm_metal_hipcub_detail::query_storage(temporary_storage_bytes);
			return 0;
		}
		const scalar_type initial_value =
			openfpm_metal_hipcub_detail::initial_scalar<scalar_type>(initial);
		const std::size_t segments =
			openfpm_metal_hipcub_detail::count_or_zero(segment_count);
		// CUB obtains the input extent from the final segment offset.  The
		// backend kernel consumes that same offset and clamps each segment.
		std::size_t input_count = 0;
		if (segments != 0)
		{
			cudaDeviceSynchronize();
			const auto * offsets = static_cast<const std::uint32_t *>(
				openfpm::metal::host_pointer_from_device(begin_offsets,
					(segments + 1) * sizeof(std::uint32_t)));
			input_count = offsets[segments];
		}
		openfpm::metal::segmented_reduce(input,input_count,begin_offsets,segments,
			output,openfpm::metal::primitive_type_of<scalar_type>::value,
			operation_traits::operation,&initial_value,value_traits::component_count);
		return 0;
	}
};

struct DeviceRadixSort
{
private:
	template<bool Descending, typename KeyInput, typename KeyOutput,
		typename ValueInput, typename ValueOutput>
	static int sort(void * temporary_storage, std::size_t & temporary_storage_bytes,
		KeyInput input_keys, KeyOutput output_keys, ValueInput input_values,
		ValueOutput output_values, int count)
	{
		using key_type = typename openfpm_metal_hipcub_detail::matching_iterator_value<
			KeyInput,KeyOutput>::type;
		using value_type = typename openfpm_metal_hipcub_detail::matching_iterator_value<
			ValueInput,ValueOutput>::type;
		static_assert(openfpm::metal::is_supported_device_primitive<key_type>::value,
			"Metal stable sort supports 32/64-bit integers and float keys");
		static_assert(std::is_trivially_copyable<value_type>::value,
			"Metal stable sort values must be trivially copyable");
		if (temporary_storage == nullptr)
		{
			openfpm_metal_hipcub_detail::query_storage(temporary_storage_bytes);
			return 0;
		}
		const std::size_t size = openfpm_metal_hipcub_detail::count_or_zero(count);
		if (size == 0) return 0;
		cudaMemcpy(output_keys,input_keys,size * sizeof(key_type),cudaMemcpyDeviceToDevice);
		cudaMemcpy(output_values,input_values,size * sizeof(value_type),cudaMemcpyDeviceToDevice);
		openfpm::metal::stable_sort_pairs(output_keys,output_values,size,
			openfpm::metal::primitive_type_of<key_type>::value,sizeof(value_type),Descending);
		return 0;
	}

public:
	template<typename KeyInput, typename KeyOutput, typename ValueInput,
		typename ValueOutput>
	static int SortPairs(void * temporary_storage, std::size_t & temporary_storage_bytes,
		KeyInput input_keys, KeyOutput output_keys, ValueInput input_values,
		ValueOutput output_values, int count)
	{
		return sort<false>(temporary_storage,temporary_storage_bytes,input_keys,
			output_keys,input_values,output_values,count);
	}

	template<typename KeyInput, typename KeyOutput, typename ValueInput,
		typename ValueOutput>
	static int SortPairsDescending(void * temporary_storage,
		std::size_t & temporary_storage_bytes, KeyInput input_keys,
		KeyOutput output_keys, ValueInput input_values, ValueOutput output_values,
		int count)
	{
		return sort<true>(temporary_storage,temporary_storage_bytes,input_keys,
			output_keys,input_values,output_values,count);
	}
};
}

// Minimal thrust merge surface used by merge_ofp.cuh.  Keeping it here makes
// the core header execute the same merge_by_key expression as CUDA and HIP.
namespace thrust
{
struct openfpm_metal_device_policy {};
constexpr openfpm_metal_device_policy device{};

template<typename AKeyIterator, typename BKeyIterator,
	typename AValueIterator, typename BValueIterator,
	typename OutputKeyIterator, typename OutputValueIterator,
	typename Comparator>
void merge_by_key(openfpm_metal_device_policy,
	AKeyIterator a_keys, AKeyIterator a_keys_end,
	BKeyIterator b_keys, BKeyIterator b_keys_end,
	AValueIterator a_values, BValueIterator b_values,
	OutputKeyIterator output_keys, OutputValueIterator output_values,
	Comparator)
{
	using a_key_type = typename openfpm_metal_hipcub_detail::iterator_value<AKeyIterator>::type;
	using b_key_type = typename openfpm_metal_hipcub_detail::iterator_value<BKeyIterator>::type;
	using output_key_type = typename openfpm_metal_hipcub_detail::iterator_value<OutputKeyIterator>::type;
	using a_value_type = typename openfpm_metal_hipcub_detail::iterator_value<AValueIterator>::type;
	using b_value_type = typename openfpm_metal_hipcub_detail::iterator_value<BValueIterator>::type;
	using output_value_type = typename openfpm_metal_hipcub_detail::iterator_value<OutputValueIterator>::type;
	static_assert(std::is_same<a_key_type,b_key_type>::value &&
		std::is_same<a_key_type,output_key_type>::value,
		"Metal merge key types must match");
	static_assert(std::is_same<a_value_type,b_value_type>::value &&
		std::is_same<a_value_type,output_value_type>::value,
		"Metal merge value types must match");
	static_assert(openfpm::metal::is_supported_device_primitive<a_key_type>::value,
		"Metal merge supports 32/64-bit integers and float keys");
	static_assert(std::is_trivially_copyable<a_value_type>::value,
		"Metal merge values must be trivially copyable");
	using ordering = openfpm_metal_hipcub_detail::comparator<Comparator,a_key_type>;
	openfpm::metal::merge_pairs(a_keys,a_values,
		static_cast<std::size_t>(a_keys_end-a_keys),b_keys,b_values,
		static_cast<std::size_t>(b_keys_end-b_keys),output_keys,output_values,
		openfpm::metal::primitive_type_of<a_key_type>::value,sizeof(a_value_type),
		ordering::descending);
}
}

#endif
