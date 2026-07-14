#include "MoltenVKContext.hpp"
#include "MoltenVKKernel.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace openfpm
{
namespace metal
{
namespace
{
void check(VkResult result, const char * operation)
{
	if (result != VK_SUCCESS)
		throw std::runtime_error(std::string(operation) + " failed with Vulkan error " +
			std::to_string(static_cast<int>(result)));
}

bool has_extension(const std::vector<VkExtensionProperties> & extensions,
	const char * name)
{
	for (const auto & extension : extensions)
		if (std::strcmp(extension.extensionName, name) == 0) return true;
	return false;
}
}

MoltenVKContext::MoltenVKContext()
{
	try
	{
	// MoltenVK otherwise enables Metal fast-math by default.  That changes
	// boundary decisions in ordinary HIP kernels (for example sqrt-based
	// cell/stencil predicates) and can make CUDA/HIP regression counts backend
	// dependent.  Prefer strict shader arithmetic for transparent semantics,
	// while still allowing an application to opt back in before first use.
	if (std::getenv("MVK_CONFIG_FAST_MATH_ENABLED") == nullptr)
		setenv("MVK_CONFIG_FAST_MATH_ENABLED","0",0);

	VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	application.pApplicationName = "OpenFPM";
	application.apiVersion = VK_API_VERSION_1_2;

	std::uint32_t instance_extension_count = 0;
	check(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr),
		"vkEnumerateInstanceExtensionProperties");
	std::vector<VkExtensionProperties> instance_extensions(instance_extension_count);
	check(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count,
		instance_extensions.data()), "vkEnumerateInstanceExtensionProperties");
	std::vector<const char *> enabled_instance_extensions;
	VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &application;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
	if (has_extension(instance_extensions,
		VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
	{
		enabled_instance_extensions.push_back(
			VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}
#endif
	instance_info.enabledExtensionCount =
		static_cast<std::uint32_t>(enabled_instance_extensions.size());
	instance_info.ppEnabledExtensionNames = enabled_instance_extensions.data();
	check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

	std::uint32_t device_count = 0;
	check(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
		"vkEnumeratePhysicalDevices");
	if (device_count == 0) throw std::runtime_error("MoltenVK found no Vulkan device");
	std::vector<VkPhysicalDevice> devices(device_count);
	check(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
		"vkEnumeratePhysicalDevices");
	physical_device_ = devices.front();

	std::uint32_t family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, nullptr);
	std::vector<VkQueueFamilyProperties> families(family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, families.data());
	bool found_compute = false;
	for (std::uint32_t i = 0; i < family_count; ++i)
	{
		if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
		{
			queue_family_ = i;
			found_compute = true;
			break;
		}
	}
	if (!found_compute) throw std::runtime_error("MoltenVK found no compute queue");

	std::uint32_t extension_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr);
	std::vector<VkExtensionProperties> available_extensions(extension_count);
	vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count,
		available_extensions.data());
	std::vector<const char *> enabled_extensions;
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
	if (has_extension(available_extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
		enabled_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

	VkPhysicalDeviceBufferDeviceAddressFeatures supported_bda{
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
	VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	supported.pNext = &supported_bda;
	vkGetPhysicalDeviceFeatures2(physical_device_, &supported);
	if (!supported_bda.bufferDeviceAddress)
		throw std::runtime_error("MoltenVK device does not support Vulkan buffer device addresses");
	if (!supported.features.shaderInt64)
		throw std::runtime_error("MoltenVK device does not support 64-bit shader integers");

	float priority = 1.0f;
	VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = queue_family_;
	queue_info.queueCount = 1;
	queue_info.pQueuePriorities = &priority;
	VkPhysicalDeviceBufferDeviceAddressFeatures bda{
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
	bda.bufferDeviceAddress = VK_TRUE;
	VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.pNext = &bda;
	VkPhysicalDeviceFeatures enabled_features{};
	enabled_features.shaderInt64 = VK_TRUE;
	device_info.pEnabledFeatures = &enabled_features;
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos = &queue_info;
	device_info.enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size());
	device_info.ppEnabledExtensionNames = enabled_extensions.data();
	check(vkCreateDevice(physical_device_, &device_info, nullptr, &device_), "vkCreateDevice");
	vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

	VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool_info.queueFamilyIndex = queue_family_;
	check(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
		"vkCreateCommandPool");
	}
	catch (...)
	{
		if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
		if (command_pool_ != VK_NULL_HANDLE)
			vkDestroyCommandPool(device_,command_pool_,nullptr);
		if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_,nullptr);
		if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_,nullptr);
		throw;
	}
}

MoltenVKContext::~MoltenVKContext()
{
	if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
	if (device_ != VK_NULL_HANDLE) destroy_moltenvk_kernel_cache(device_);
	if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
	if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
	if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

void MoltenVKContext::synchronize() const
{
	std::lock_guard<std::recursive_mutex> lock(execution_mutex_);
	check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
}

MoltenVKContext & moltenvk_context()
{
	static MoltenVKContext context;
	return context;
}

void metal_synchronize()
{
	moltenvk_context().synchronize();
}

} // namespace metal
} // namespace openfpm
