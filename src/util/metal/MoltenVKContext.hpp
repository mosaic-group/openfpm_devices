#ifndef OPENFPM_MOLTENVK_CONTEXT_HPP_
#define OPENFPM_MOLTENVK_CONTEXT_HPP_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>

namespace openfpm
{
namespace metal
{

class MoltenVKContext
{
	VkInstance instance_ = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
	VkDevice device_ = VK_NULL_HANDLE;
	VkQueue queue_ = VK_NULL_HANDLE;
	VkCommandPool command_pool_ = VK_NULL_HANDLE;
	std::uint32_t queue_family_ = 0;
	mutable std::recursive_mutex execution_mutex_;

public:
	MoltenVKContext();
	~MoltenVKContext();
	MoltenVKContext(const MoltenVKContext &) = delete;
	MoltenVKContext & operator=(const MoltenVKContext &) = delete;

	VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
	VkDevice device() const noexcept { return device_; }
	VkQueue queue() const noexcept { return queue_; }
	VkCommandPool command_pool() const noexcept { return command_pool_; }
	std::uint32_t queue_family() const noexcept { return queue_family_; }
	std::recursive_mutex & execution_mutex() const noexcept
	{return execution_mutex_;}
	void synchronize() const;
};

MoltenVKContext & moltenvk_context();
void metal_synchronize();

} // namespace metal
} // namespace openfpm

// Keep the existing cudify-facing spelling while the backend identifier stays
// CUDA_ON_BACKEND=METAL.
inline void metal_synchronize() { openfpm::metal::metal_synchronize(); }

#endif
