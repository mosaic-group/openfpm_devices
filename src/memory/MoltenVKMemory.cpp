#include "MoltenVKMemory.hpp"

#include "util/metal/MoltenVKContext.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace
{
constexpr std::size_t cuda_allocation_guard = 32;

std::size_t allocation_bytes(std::size_t logical_size)
{
	return logical_size == 0 ? 0 : logical_size + cuda_allocation_guard;
}

struct mapped_allocation
{
	VkDeviceAddress address;
	std::size_t size;
	void * mapped;
	VkDeviceMemory memory;
	bool coherent;
};

std::vector<mapped_allocation> & mapped_allocations()
{
	static std::vector<mapped_allocation> allocations;
	return allocations;
}

std::mutex & mapped_allocations_mutex()
{
	static std::mutex mutex;
	return mutex;
}

void register_mapping(VkDeviceAddress address, std::size_t size, void * mapped,
	VkDeviceMemory memory, bool coherent)
{
	std::lock_guard<std::mutex> lock(mapped_allocations_mutex());
	mapped_allocations().push_back({address,size,mapped,memory,coherent});
}

void unregister_mapping(VkDeviceAddress address)
{
	std::lock_guard<std::mutex> lock(mapped_allocations_mutex());
	auto & allocations = mapped_allocations();
	allocations.erase(std::remove_if(allocations.begin(),allocations.end(),
		[address](const mapped_allocation & allocation)
		{return allocation.address == address;}),allocations.end());
}

std::uint32_t memory_type(VkPhysicalDevice physical_device, std::uint32_t bits,
	VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, bool & coherent)
{
	VkPhysicalDeviceMemoryProperties properties{};
	vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
	for (int pass = 0; pass < 2; ++pass)
	{
		const VkMemoryPropertyFlags wanted = pass == 0 ? required | preferred : required;
		for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i)
		{
			if ((bits & (1u << i)) != 0 &&
				(properties.memoryTypes[i].propertyFlags & wanted) == wanted)
			{
				coherent = (properties.memoryTypes[i].propertyFlags &
					VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
				return i;
			}
		}
	}
	throw std::runtime_error("MoltenVK has no host-visible memory type");
}

void check(VkResult result, const char * operation)
{
	if (result != VK_SUCCESS)
		throw std::runtime_error(std::string(operation) + " failed with Vulkan error " +
			std::to_string(static_cast<int>(result)));
}

template<typename Record>
void submit_transfer(Record && record)
{
	auto & context = openfpm::metal::moltenvk_context();
	std::lock_guard<std::recursive_mutex> execution_lock(
		context.execution_mutex());
	VkCommandBuffer command = VK_NULL_HANDLE;
	try
	{
		VkCommandBufferAllocateInfo allocation{
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		allocation.commandPool = context.command_pool();
		allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocation.commandBufferCount = 1;
		check(vkAllocateCommandBuffers(context.device(),&allocation,&command),
			"vkAllocateCommandBuffers");
		VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		check(vkBeginCommandBuffer(command,&begin),"vkBeginCommandBuffer");
		record(command);
		check(vkEndCommandBuffer(command),"vkEndCommandBuffer");
		VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &command;
		check(vkQueueSubmit(context.queue(),1,&submit,VK_NULL_HANDLE),
			"vkQueueSubmit");
		context.synchronize();
	}
	catch (...)
	{
		if (command != VK_NULL_HANDLE)
			vkFreeCommandBuffers(context.device(),context.command_pool(),1,&command);
		throw;
	}
	vkFreeCommandBuffers(context.device(),context.command_pool(),1,&command);
}
}

MoltenVKMemory::MoltenVKMemory(std::size_t size)
{
	if (!allocate(size)) throw std::runtime_error("MoltenVK allocation failed");
}

MoltenVKMemory::MoltenVKMemory(const MoltenVKMemory & other)
{
	if (!allocate(other.size_) || !copy(other))
		throw std::runtime_error("MoltenVK copy construction failed");
}

MoltenVKMemory::MoltenVKMemory(MoltenVKMemory && other) noexcept
	: buffer_(std::exchange(other.buffer_,VK_NULL_HANDLE)),
	  memory_(std::exchange(other.memory_,VK_NULL_HANDLE)),
	  mapped_(std::exchange(other.mapped_,nullptr)),
	  host_(std::exchange(other.host_,nullptr)),
	  device_address_(std::exchange(other.device_address_,0)),
	  size_(std::exchange(other.size_,0)),
	  ref_count_(std::exchange(other.ref_count_,0)),
	  coherent_(std::exchange(other.coherent_,true))
{}

MoltenVKMemory & MoltenVKMemory::operator=(const MoltenVKMemory & other)
{
	if (this == &other) return *this;
	if (size_ < other.size_ && !resize(other.size_))
		throw std::runtime_error("MoltenVK resize failed");
	if (!copy(other)) throw std::runtime_error("MoltenVK copy failed");
	return *this;
}

MoltenVKMemory & MoltenVKMemory::operator=(MoltenVKMemory && other) noexcept
{
	if (this != &other)
	{
		destroy();
		buffer_ = std::exchange(other.buffer_,VK_NULL_HANDLE);
		memory_ = std::exchange(other.memory_,VK_NULL_HANDLE);
		mapped_ = std::exchange(other.mapped_,nullptr);
		host_ = std::exchange(other.host_,nullptr);
		device_address_ = std::exchange(other.device_address_,0);
		size_ = std::exchange(other.size_,0);
		ref_count_ = std::exchange(other.ref_count_,0);
		coherent_ = std::exchange(other.coherent_,true);
	}
	return *this;
}

MoltenVKMemory::~MoltenVKMemory()
{
	// Match the memory interface ownership contract used by memory_c. Views
	// retain an allocator with incRef(); a temporary allocator object must not
	// release its Vulkan allocation while one of those views is still alive.
	if (ref_count_ == 0) destroy();
}

bool MoltenVKMemory::allocate(std::size_t size)
{
	if (buffer_ != VK_NULL_HANDLE) return size == size_;
	if (size == 0) return true;
	try
	{
		auto & context = openfpm::metal::moltenvk_context();
		VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			buffer_info.size = allocation_bytes(size);
		buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		check(vkCreateBuffer(context.device(), &buffer_info, nullptr, &buffer_),
			"vkCreateBuffer");

		VkMemoryRequirements requirements{};
		vkGetBufferMemoryRequirements(context.device(), buffer_, &requirements);
		VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
		flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		allocation.pNext = &flags;
		allocation.allocationSize = requirements.size;
		allocation.memoryTypeIndex = memory_type(context.physical_device(),
			requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, coherent_);
		check(vkAllocateMemory(context.device(), &allocation, nullptr, &memory_),
			"vkAllocateMemory");
		check(vkBindBufferMemory(context.device(), buffer_, memory_, 0),
			"vkBindBufferMemory");
			check(vkMapMemory(context.device(), memory_, 0, allocation_bytes(size), 0,
				&mapped_), "vkMapMemory");
		VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
		address_info.buffer = buffer_;
		device_address_ = vkGetBufferDeviceAddress(context.device(), &address_info);
		if (device_address_ == 0) throw std::runtime_error("vkGetBufferDeviceAddress returned zero");
		size_ = size;
			register_mapping(device_address_,allocation_bytes(size_),mapped_,memory_,
				coherent_);
		if (std::getenv("OPENFPM_METAL_TRACE_MEMORY") != nullptr)
			std::fprintf(stderr,
				"MoltenVK allocation object=%p address=0x%llx size=%zu mapped=%p coherent=%d\n",
				static_cast<void *>(this),
				static_cast<unsigned long long>(device_address_),size_,mapped_,
				coherent_ ? 1 : 0);
		return true;
	}
	catch (const std::exception & error)
	{
		std::fprintf(stderr, "MoltenVKMemory::allocate: %s\n", error.what());
		destroy();
		return false;
	}
}

bool MoltenVKMemory::resize(std::size_t size)
{
	if (size <= size_) return true;
	MoltenVKMemory replacement;
	if (!replacement.allocate(size)) return false;
	if (size_ != 0 && !replacement.copyDeviceToDevice(*this)) return false;
	if (host_ != nullptr)
	{
		replacement.allocate_host();
		std::memcpy(replacement.host_,host_,allocation_bytes(size_));
	}
	// References retain this allocator object, not a particular VkBuffer. Keep
	// its ownership count attached to the object while exchanging allocations.
	// Otherwise an externally retained vector can observe ref()==0 during
	// resize and delete its still-live allocator.
	const std::size_t retained_references = ref_count_;
	swap(replacement);
	ref_count_ = retained_references;
	replacement.ref_count_ = 0;
	return true;
}

void MoltenVKMemory::destroy()
{
	if (buffer_ != VK_NULL_HANDLE || memory_ != VK_NULL_HANDLE)
	{
		auto & context = openfpm::metal::moltenvk_context();
		context.synchronize();
		if (device_address_ != 0) unregister_mapping(device_address_);
		if (mapped_ != nullptr) vkUnmapMemory(context.device(), memory_);
		if (buffer_ != VK_NULL_HANDLE)
			vkDestroyBuffer(context.device(), buffer_, nullptr);
		if (memory_ != VK_NULL_HANDLE) vkFreeMemory(context.device(), memory_, nullptr);
	}
	std::free(host_);
	buffer_ = VK_NULL_HANDLE;
	memory_ = VK_NULL_HANDLE;
	mapped_ = nullptr;
	host_ = nullptr;
	device_address_ = 0;
	size_ = 0;
}

bool MoltenVKMemory::copy(const memory & other)
{
	if (other.size() > size_) return false;
	if (auto source = dynamic_cast<const MoltenVKMemory *>(&other))
		return copyDeviceToDevice(*source);
	if (other.size() != 0)
	{
		allocate_host();
		std::memcpy(host_,other.getPointer(),other.size());
	}
	return true;
}

bool MoltenVKMemory::copyDeviceToDevice(const MoltenVKMemory & other)
{
	if (other.size_ > size_ || (other.size_ != 0 && mapped_ == nullptr))
		return false;
	if (other.size_ == 0 || this == &other) return true;
	const std::size_t bytes = allocation_bytes(other.size_);
	// Vulkan buffer copies stay device-resident. The uncommon byte-sized tail
	// case retains the coherent-memory path because vkCmdCopyBuffer requires
	// four-byte transfer granularity on the supported MoltenVK devices.
	if ((bytes & 3u) == 0)
	{
		VkBufferCopy region{0,0,bytes};
		submit_transfer([&](VkCommandBuffer command)
		{
			vkCmdCopyBuffer(command,other.buffer_,buffer_,1,&region);
		});
		return true;
	}
	openfpm::metal::metal_synchronize();
	std::memcpy(mapped_,other.mapped_,bytes);
	openfpm::metal::flush_device_pointer(getDevicePointer(),bytes);
	return true;
}

void MoltenVKMemory::deviceToDevice(void * source, std::size_t start,
	std::size_t stop, std::size_t destination_offset)
{
	if (start > stop || destination_offset > size_ ||
		stop - start > size_ - destination_offset)
		throw std::out_of_range("MoltenVK device-to-device range");
	const std::size_t bytes = stop - start;
	if (bytes == 0) return;
	openfpm::metal::metal_synchronize();
	const auto source_address = reinterpret_cast<const void *>(
		reinterpret_cast<std::uintptr_t>(source) + start);
	const void * mapped_source =
		openfpm::metal::host_pointer_from_device(source_address,bytes);
	std::memmove(static_cast<unsigned char *>(mapped_) + destination_offset,
		mapped_source,bytes);
	const auto destination_address = reinterpret_cast<const void *>(
		static_cast<std::uintptr_t>(device_address_) + destination_offset);
	openfpm::metal::flush_device_pointer(destination_address,bytes);
}

void * MoltenVKMemory::getDevicePointer()
{
	return reinterpret_cast<void *>(static_cast<std::uintptr_t>(device_address_));
}

void MoltenVKMemory::allocate_host() const
{
	if (host_ != nullptr || size_ == 0) return;
	host_ = std::malloc(allocation_bytes(size_));
	if (host_ == nullptr) throw std::bad_alloc();
}

void * MoltenVKMemory::getPointer()
{
	allocate_host();
	return host_;
}

const void * MoltenVKMemory::getPointer() const
{
	allocate_host();
	return host_;
}

void MoltenVKMemory::hostToDevice()
{
	if (size_ == 0) return;
	allocate_host();
	openfpm::metal::metal_synchronize();
	std::memcpy(mapped_,host_,allocation_bytes(size_));
	if (!coherent_ && memory_ != VK_NULL_HANDLE)
	{
		VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		range.memory = memory_;
		range.size = VK_WHOLE_SIZE;
		check(vkFlushMappedMemoryRanges(openfpm::metal::moltenvk_context().device(), 1, &range),
			"vkFlushMappedMemoryRanges");
	}
}

void MoltenVKMemory::deviceToHost()
{
	if (memory_ == VK_NULL_HANDLE) return;
	if (std::getenv("OPENFPM_METAL_TRACE_MEMORY") != nullptr)
		std::fprintf(stderr,
			"MoltenVK deviceToHost object=%p address=0x%llx size=%zu coherent=%d\n",
			static_cast<void *>(this),
			static_cast<unsigned long long>(device_address_),size_,
			coherent_ ? 1 : 0);
	openfpm::metal::metal_synchronize();
	if (!coherent_)
	{
		VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		range.memory = memory_;
		range.size = VK_WHOLE_SIZE;
		check(vkInvalidateMappedMemoryRanges(openfpm::metal::moltenvk_context().device(), 1,
			&range), "vkInvalidateMappedMemoryRanges");
	}
	allocate_host();
	std::memcpy(host_,mapped_,allocation_bytes(size_));
}

void MoltenVKMemory::hostToDevice(MoltenVKMemory & source)
{
	if (source.size_ == 0) return;
	if (size_ < source.size_ && !resize(source.size_))
		throw std::runtime_error("MoltenVK host-to-device resize failed");
	source.allocate_host();
	openfpm::metal::metal_synchronize();
	std::memcpy(mapped_,source.host_,allocation_bytes(source.size_));
	if (!coherent_)
	{
		VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		range.memory = memory_;
		range.size = VK_WHOLE_SIZE;
		check(vkFlushMappedMemoryRanges(
			openfpm::metal::moltenvk_context().device(),1,&range),
			"vkFlushMappedMemoryRanges");
	}
}

void MoltenVKMemory::deviceToHost(MoltenVKMemory & destination)
{
	if (size_ == 0) return;
	if (destination.size_ < size_ && !destination.resize(size_))
		throw std::runtime_error("MoltenVK device-to-host resize failed");
	// Match cudaMemcpy(destination.hm, this->dm, ...): this overload must not
	// overwrite this object's independent host shadow.  Nested-vector teardown
	// uses a temporary destination to inspect and destroy the device-side kernel
	// views while retaining the original host-side C++ objects for their normal
	// destructors.
	openfpm::metal::metal_synchronize();
	if (!coherent_)
	{
		VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		range.memory = memory_;
		range.size = VK_WHOLE_SIZE;
		check(vkInvalidateMappedMemoryRanges(
			openfpm::metal::moltenvk_context().device(),1,&range),
			"vkInvalidateMappedMemoryRanges");
	}
	destination.allocate_host();
	std::memcpy(destination.host_,mapped_,allocation_bytes(size_));
}

void MoltenVKMemory::hostToDevice(std::size_t start, std::size_t stop)
{
	if (start > stop || stop > size_) throw std::out_of_range("hostToDevice range");
	if (start == stop) return;
	allocate_host();
	openfpm::metal::metal_synchronize();
	std::memcpy(static_cast<unsigned char *>(mapped_) + start,
		static_cast<unsigned char *>(host_) + start,stop - start);
	if (!coherent_)
	{
		VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		range.memory = memory_;
		range.size = VK_WHOLE_SIZE;
		check(vkFlushMappedMemoryRanges(
			openfpm::metal::moltenvk_context().device(),1,&range),
			"vkFlushMappedMemoryRanges");
	}
}

void MoltenVKMemory::deviceToHost(std::size_t start, std::size_t stop)
{
	if (start > stop || stop > size_) throw std::out_of_range("deviceToHost range");
	if (start == stop) return;
	openfpm::metal::metal_synchronize();
	if (!coherent_)
	{
		VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		range.memory = memory_;
		range.size = VK_WHOLE_SIZE;
		check(vkInvalidateMappedMemoryRanges(
			openfpm::metal::moltenvk_context().device(),1,&range),
			"vkInvalidateMappedMemoryRanges");
	}
	allocate_host();
	std::memcpy(static_cast<unsigned char *>(host_) + start,
		static_cast<unsigned char *>(mapped_) + start,stop - start);
}

void MoltenVKMemory::fill(unsigned char value)
{
	if (size_ == 0) return;
	if ((size_ & 3u) == 0)
	{
		const std::uint32_t pattern = static_cast<std::uint32_t>(value) * 0x01010101u;
		submit_transfer([&](VkCommandBuffer command)
		{
			vkCmdFillBuffer(command,buffer_,0,size_,pattern);
		});
	}
	else
	{
		openfpm::metal::metal_synchronize();
		std::memset(mapped_,value,size_);
		if (!coherent_)
		{
			VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
			range.memory = memory_;
			range.size = VK_WHOLE_SIZE;
			check(vkFlushMappedMemoryRanges(
				openfpm::metal::moltenvk_context().device(),1,&range),
				"vkFlushMappedMemoryRanges");
		}
	}
	if (host_ != nullptr) std::memset(host_,value,size_);
}

void MoltenVKMemory::swap(MoltenVKMemory & other) noexcept
{
	// This deliberately mirrors CudaMemory::swap rather than std::swap.  The
	// allocator objects are retained by memory_c; swap_nomode exchanges their
	// allocations while leaving the externally-owned allocator on each side.
	// The destination adopts the source retention count, but changing the
	// source count would let a temporary memory_c delete a global scratch
	// allocator (exp_tmp/exp_tmp2) when it goes out of scope.
	using std::swap;
	swap(buffer_, other.buffer_);
	swap(memory_, other.memory_);
	swap(mapped_, other.mapped_);
	swap(host_, other.host_);
	swap(device_address_, other.device_address_);
	swap(size_, other.size_);
	ref_count_ = other.ref_count_;
	swap(coherent_, other.coherent_);
}

namespace openfpm
{
namespace metal
{
void * host_pointer_from_device(const void * device_pointer, std::size_t bytes)
{
	const auto address = static_cast<VkDeviceAddress>(
		reinterpret_cast<std::uintptr_t>(device_pointer));
	std::lock_guard<std::mutex> lock(mapped_allocations_mutex());
	for (const auto & allocation : mapped_allocations())
	{
		if (address < allocation.address) continue;
		const std::size_t offset = static_cast<std::size_t>(address - allocation.address);
		if (offset <= allocation.size && bytes <= allocation.size - offset)
			return static_cast<unsigned char *>(allocation.mapped) + offset;
	}
	throw std::out_of_range("pointer is not part of a MoltenVK allocation");
}

void flush_device_pointer(const void * device_pointer, std::size_t bytes)
{
	const auto address = static_cast<VkDeviceAddress>(
		reinterpret_cast<std::uintptr_t>(device_pointer));
	std::lock_guard<std::mutex> lock(mapped_allocations_mutex());
	for (const auto & allocation : mapped_allocations())
	{
		if (address < allocation.address) continue;
		const std::size_t offset = static_cast<std::size_t>(address - allocation.address);
		if (offset > allocation.size || bytes > allocation.size - offset) continue;
		if (!allocation.coherent)
		{
			VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
			range.memory = allocation.memory;
			range.size = VK_WHOLE_SIZE;
			check(vkFlushMappedMemoryRanges(moltenvk_context().device(),1,&range),
				"vkFlushMappedMemoryRanges");
		}
		return;
	}
	throw std::out_of_range("pointer is not part of a MoltenVK allocation");
}
}
}
