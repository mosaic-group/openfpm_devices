#ifndef OPENFPM_MOLTENVK_MEMORY_HPP_
#define OPENFPM_MOLTENVK_MEMORY_HPP_

#include "memory.hpp"
#include "util/cuda_util.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>

__device__ inline std::size_t align_number_device(std::size_t alignment,
	std::size_t number)
{
	return number + ((number % alignment) != 0) *
		(alignment - number % alignment);
}

// Shared CUDA/HIP kernel views use this symbol for optional bounds diagnostics.
// Keep the compatibility symbol in the selected device-memory header, just as
// CudaMemory does for native CUDA/HIP builds.
static __device__ unsigned char global_cuda_error_array[256];

class MoltenVKMemory : public memory
{
	VkBuffer buffer_ = VK_NULL_HANDLE;
	VkDeviceMemory memory_ = VK_NULL_HANDLE;
	void * mapped_ = nullptr;
	mutable void * host_ = nullptr;
	VkDeviceAddress device_address_ = 0;
	std::size_t size_ = 0;
	std::size_t ref_count_ = 0;
	bool coherent_ = true;

	void allocate_host() const;

public:
	MoltenVKMemory() = default;
	explicit MoltenVKMemory(std::size_t size);
	MoltenVKMemory(const MoltenVKMemory & other);
	MoltenVKMemory(MoltenVKMemory && other) noexcept;
	MoltenVKMemory & operator=(const MoltenVKMemory & other);
	MoltenVKMemory & operator=(MoltenVKMemory && other) noexcept;
	~MoltenVKMemory() override;

	bool allocate(std::size_t size) override;
	bool resize(std::size_t size) override;
	void destroy() override;
	bool copy(const memory & other) override;
	bool copyDeviceToDevice(const MoltenVKMemory & other);
	void deviceToDevice(void * source, std::size_t start, std::size_t stop,
		std::size_t destination_offset);
	std::size_t size() const override { return size_; }
	void * getPointer() override;
	const void * getPointer() const override;
	void * getDevicePointer() override;
	void hostToDevice() override;
	void deviceToHost() override;
	void hostToDevice(MoltenVKMemory & source);
	void deviceToHost(MoltenVKMemory & destination);
	void hostToDevice(std::size_t start, std::size_t stop) override;
	void deviceToHost(std::size_t start, std::size_t stop) override;
	void fill(unsigned char value) override;
	void incRef() override { ++ref_count_; }
	void decRef() override { if (ref_count_ != 0) --ref_count_; }
	long int ref() override { return static_cast<long int>(ref_count_); }
	// In the memory interface this answers whether objects in newly allocated
	// storage have already been C++-constructed, not whether a buffer exists.
	// Device allocators return false so memory_c performs placement construction.
	bool isInitialized() override { return false; }
	bool flush() override { hostToDevice(); return true; }
	void isNotSync() {}
	void swap(MoltenVKMemory & other) noexcept;

	VkBuffer vulkan_buffer() const noexcept { return buffer_; }
	VkDeviceAddress vulkan_device_address() const noexcept { return device_address_; }
	static constexpr bool isDeviceHostSame() { return false; }
	void * toKernel() { return getDevicePointer(); }
};

namespace openfpm
{
namespace metal
{
/** Resolve a Vulkan buffer device address to its persistently mapped CPU address. */
void * host_pointer_from_device(const void * device_pointer, std::size_t bytes = 0);

/** Publish CPU writes made through host_pointer_from_device to the Vulkan device. */
void flush_device_pointer(const void * device_pointer, std::size_t bytes = 0);
}
}

#endif
