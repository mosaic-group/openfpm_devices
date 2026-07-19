#ifndef OPENFPM_MOLTENVK_KERNEL_HPP_
#define OPENFPM_MOLTENVK_KERNEL_HPP_

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

namespace openfpm
{
namespace metal
{

inline VkExtent3D to_extent(unsigned int x) { return VkExtent3D{x, 1, 1}; }

template<typename Dim>
VkExtent3D to_extent(const Dim & value)
{
	return VkExtent3D{static_cast<std::uint32_t>(value.x),
		static_cast<std::uint32_t>(value.y), static_cast<std::uint32_t>(value.z)};
}

/** Launch clspv-generated Vulkan SPIR-V using POD push-constant arguments. */
void launch_spirv(const std::uint32_t * words, std::size_t word_count,
	const char * entry_point, const void * arguments, std::size_t argument_size,
	VkExtent3D workgroups, VkExtent3D threads);

struct KernelArgumentLayout
{
	std::uint32_t offset;
	std::uint32_t size;
	std::uint32_t alignment;
};

struct KernelArgumentView
{
	const void * data;
	std::size_t size;
	std::size_t alignment;
};

/** clspv-reflected byte offsets for the push-constant launch ABI. */
struct KernelPushConstantLayout
{
	std::uint32_t size;
	std::int32_t argument_pointer;
	std::int32_t global_offset;
	std::int32_t enqueued_local_size;
	std::int32_t global_size;
	std::int32_t region_offset;
	std::int32_t num_workgroups;
	std::int32_t region_group_offset;
};

/** Register one ABI-normalized SPIR-V kernel and its compiler-generated ABI. */
void register_spirv_kernel(const std::uint32_t * words, std::size_t word_count,
	const char * entry_point, std::size_t packed_argument_size,
	std::size_t packed_argument_alignment,
	KernelPushConstantLayout push_constants,
	const KernelArgumentLayout * argument_layouts,
	std::size_t argument_count);

void launch_registered_spirv(const char * requested_entry,
	const KernelArgumentView * arguments, std::size_t argument_count,
	VkExtent3D workgroups, VkExtent3D threads);

/** Bind the exact host kernel handle emitted by HIP clang to its SPIR-V name. */
void register_kernel_handle(const void * host_function,
	const char * device_entry_point);

/** Launch using the exact HIP compiler handle and its void** argument array. */
void launch_registered_handle(const void * host_function,
	void * const * arguments, VkExtent3D workgroups, VkExtent3D threads);

/**
 * Launch a named kernel from an ordinary C++ host translation unit.
 *
 * HIP clang registers the exact host stub address through
 * __hipRegisterFunction.  Header-defined OpenFPM kernels have the same weak
 * Itanium symbol in regular C++ translation units, so using that function
 * address avoids ambiguous source-name matching when several template
 * instantiations share one CUDA_LAUNCH spelling.
 */
template<typename Function, typename... Args>
void launch_registered_function(Function function, VkExtent3D workgroups,
	VkExtent3D threads, const Args &... args)
{
	static_assert(std::is_pointer<Function>::value &&
		std::is_function<typename std::remove_pointer<Function>::type>::value,
		"Metal launch requires a named HIP/CUDA kernel function");
	std::array<void *,sizeof...(Args)> pointers{{
		const_cast<void *>(static_cast<const void *>(std::addressof(args)))...}};
	launch_registered_handle(reinterpret_cast<const void *>(function),
		sizeof...(Args) == 0 ? nullptr : pointers.data(),workgroups,threads);
}

/** Release cached shader modules and pipelines before the Vulkan device dies. */
void destroy_moltenvk_kernel_cache(VkDevice device) noexcept;

template<typename... Args>
void launch_registered(const char * requested_entry, VkExtent3D workgroups,
    VkExtent3D threads, const Args &... args)
{
	static_assert((!std::is_polymorphic<typename std::decay<Args>::type>::value && ...),
		"Polymorphic objects cannot be Metal kernel arguments");
	const std::array<KernelArgumentView,sizeof...(Args)> views{{
		KernelArgumentView{std::addressof(args),sizeof(Args),alignof(Args)}...}};
	launch_registered_spirv(requested_entry,views.data(),views.size(),
		workgroups,threads);
}

} // namespace metal
} // namespace openfpm

#endif
