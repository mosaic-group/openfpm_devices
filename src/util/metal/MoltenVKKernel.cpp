#include "MoltenVKKernel.hpp"

#include "MoltenVKContext.hpp"
#include "memory/MoltenVKMemory.hpp"

#include <array>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace openfpm
{
namespace metal
{
namespace
{
struct RegisteredKernel
{
	std::vector<std::uint32_t> words;
	std::string entry;
	std::size_t packed_argument_size = 0;
	std::size_t packed_argument_alignment = 1;
	KernelPushConstantLayout push_constants{};
	std::vector<KernelArgumentLayout> argument_layouts;
};

struct CachedPipeline
{
	VkShaderModule module = VK_NULL_HANDLE;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
};

std::unordered_map<std::string,CachedPipeline> & pipeline_cache()
{
	// The Vulkan context explicitly destroys these handles before its device.
	// Keeping the map itself process-lifetime avoids cross-translation-unit
	// static-destruction ordering hazards.
	static auto * value = new std::unordered_map<std::string,CachedPipeline>;
	return *value;
}

std::string pipeline_cache_key(const RegisteredKernel & kernel, VkExtent3D threads)
{
	// Mangled names are normally unique, but separate translation units may
	// legitimately instantiate the same name. Include the module contents so a
	// cached pipeline can never be reused with a different SPIR-V body.
	std::size_t module_hash = 1469598103934665603ull;
	for (const std::uint32_t word : kernel.words)
	{
		module_hash ^= word;
		module_hash *= 1099511628211ull;
	}
	return kernel.entry + ':' + std::to_string(module_hash) + ':' +
		std::to_string(threads.width) + ':' +
		std::to_string(threads.height) + ':' + std::to_string(threads.depth);
}

std::vector<RegisteredKernel> & kernels()
{
	static std::vector<RegisteredKernel> value;
	return value;
}

std::mutex & module_mutex()
{
	static std::mutex value;
	return value;
}

std::unordered_map<const void *,std::string> & kernel_handles()
{
	static std::unordered_map<const void *,std::string> value;
	return value;
}

void check(VkResult result, const char * operation)
{
	if (result != VK_SUCCESS)
		throw std::runtime_error(std::string(operation) + " failed with Vulkan error " +
			std::to_string(static_cast<int>(result)));
}

std::string source_kernel_name(const char * requested)
{
	std::string name(requested == nullptr ? "" : requested);
	const auto first = name.find_first_not_of(" \t(");
	if (first == std::string::npos) return {};
	name.erase(0,first);
	const auto template_start = name.find('<');
	const auto paren = name.find('(');
	const auto end = std::min(template_start == std::string::npos ? name.size() : template_start,
		paren == std::string::npos ? name.size() : paren);
	name.resize(end);
	while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())))
		name.pop_back();
	const auto scope = name.rfind("::");
	if (scope != std::string::npos) name.erase(0,scope + 2);
	return name;
}

bool entry_matches(const std::string & entry, const std::string & requested_name)
{
	if (entry == requested_name) return true;
	// Match an Itanium source-name component. This handles both top-level
	// (_Z14vector_add...) and namespace-nested (_ZN7openfpm...21kernel_name...)
	// entries without requiring a platform C++ demangler at runtime.
	if (entry.size() < 4 || entry[0] != '_' || entry[1] != 'Z') return false;
	std::size_t position = entry.find(requested_name);
	while (position != std::string::npos)
	{
		std::size_t digits = position;
		while (digits > 2 && std::isdigit(static_cast<unsigned char>(entry[digits - 1])))
			--digits;
		std::size_t length = 0;
		for (std::size_t i = digits; i < position; ++i)
			length = length * 10 + static_cast<std::size_t>(entry[i] - '0');
		if (digits != position && length == requested_name.size()) return true;
		position = entry.find(requested_name,position + 1);
	}
	return false;
}

} // namespace

void register_spirv_kernel(const std::uint32_t * words, std::size_t word_count,
	const char * entry_point, std::size_t packed_argument_size,
	std::size_t packed_argument_alignment,
	KernelPushConstantLayout push_constants,
	const KernelArgumentLayout * argument_layouts,
	std::size_t argument_count)
{
	if (words == nullptr || word_count == 0 || entry_point == nullptr)
		throw std::invalid_argument("cannot register an empty SPIR-V module");
	RegisteredKernel kernel;
	kernel.words.assign(words, words + word_count);
	kernel.entry = entry_point;
	kernel.packed_argument_size = packed_argument_size;
	kernel.packed_argument_alignment = packed_argument_alignment;
	kernel.push_constants = push_constants;
	// clspv legitimately omits the argument-block push constant when all kernel
	// arguments disappear during optimization (for example a diagnostics-only
	// kernel in a non-SE_CLASS build). Its reflected SPIR-V layout is the
	// authoritative runtime contract; a nonzero source ABI record alone does
	// not imply that the shader still consumes that record.
	if (push_constants.argument_pointer >= 0 &&
		static_cast<std::uint32_t>(push_constants.argument_pointer) +
			sizeof(std::uint64_t) > push_constants.size)
		throw std::invalid_argument("invalid Metal push-constant pointer ABI");
	for (const std::int32_t offset : {push_constants.global_offset,
		push_constants.enqueued_local_size,push_constants.global_size,
		push_constants.region_offset,push_constants.num_workgroups,
		push_constants.region_group_offset})
		if (offset >= 0 && static_cast<std::uint32_t>(offset) + 12u >
			push_constants.size)
			throw std::invalid_argument("invalid Metal push-constant builtin ABI");
	if (argument_count != 0 && argument_layouts == nullptr)
		throw std::invalid_argument("missing Metal kernel argument layout");
	kernel.argument_layouts.assign(argument_layouts,argument_layouts + argument_count);
	for (const auto & layout : kernel.argument_layouts)
		if (layout.offset + layout.size > packed_argument_size)
			throw std::invalid_argument("invalid Metal kernel argument layout");
	std::lock_guard<std::mutex> lock(module_mutex());
	kernels().push_back(std::move(kernel));
}

void register_kernel_handle(const void * host_function,
	const char * device_entry_point)
{
	if (host_function == nullptr || device_entry_point == nullptr)
		throw std::invalid_argument("invalid HIP Metal kernel registration");
	std::lock_guard<std::mutex> lock(module_mutex());
	auto result = kernel_handles().emplace(host_function,device_entry_point);
	if (!result.second && result.first->second != device_entry_point)
		throw std::runtime_error("HIP kernel handle registered with two device entries");
}

void launch_registered_handle(const void * host_function,
	void * const * arguments, VkExtent3D workgroups, VkExtent3D threads)
{
	std::string entry;
	std::vector<KernelArgumentLayout> layouts;
	{
		std::lock_guard<std::mutex> lock(module_mutex());
		const auto handle = kernel_handles().find(host_function);
		if (handle == kernel_handles().end())
			throw std::runtime_error("HIP kernel handle is not registered");
		entry = handle->second;
		for (const auto & kernel : kernels())
			if (kernel.entry == entry)
			{
				layouts = kernel.argument_layouts;
				break;
			}
	}
	if (layouts.empty() && arguments != nullptr)
		throw std::runtime_error("SPIR-V ABI is not registered for HIP kernel " + entry);
	if (!layouts.empty() && arguments == nullptr)
		throw std::runtime_error("HIP launch omitted its kernel arguments");
	std::vector<KernelArgumentView> views;
	views.reserve(layouts.size());
	for (std::size_t i = 0; i < layouts.size(); ++i)
		views.push_back({arguments[i],layouts[i].size,layouts[i].alignment});
	launch_registered_spirv(entry.c_str(),views.data(),views.size(),
		workgroups,threads);
}

void launch_registered_spirv(const char * requested_entry,
	const KernelArgumentView * arguments, std::size_t argument_count,
	VkExtent3D workgroups,
	VkExtent3D threads)
{
	if (workgroups.width == 0 || workgroups.height == 0 || workgroups.depth == 0)
		return;
	RegisteredKernel selected;
	bool found = false;
	const std::string requested_name = source_kernel_name(requested_entry);
	{
		std::lock_guard<std::mutex> lock(module_mutex());
		for (const auto & kernel : kernels())
		{
			if (entry_matches(kernel.entry,requested_name))
			{
				if (found && selected.entry != kernel.entry)
					throw std::runtime_error("ambiguous Metal kernel entry: " +
						std::string(requested_entry));
				selected = kernel;
				found = true;
			}
		}
		// A translation unit with one instantiated kernel has an unambiguous
		// mangled entry even when CUDA_LAUNCH contains only its source spelling.
		if (!found && kernels().size() == 1)
		{
			selected = kernels().front();
			found = true;
		}
	}
	if (!found)
		throw std::runtime_error("Metal kernel is not registered: " +
			std::string(requested_entry));
	if (argument_count != selected.argument_layouts.size())
		throw std::runtime_error("Metal kernel argument-count mismatch for " +
			requested_name + ": host supplied " + std::to_string(argument_count) +
			", device ABI requires " +
			std::to_string(selected.argument_layouts.size()));
	if (std::getenv("OPENFPM_METAL_TRACE_KERNELS") != nullptr)
		std::fprintf(stderr,"Metal HIP launch: %s grid=(%u,%u,%u) block=(%u,%u,%u)\n",
			selected.entry.c_str(),workgroups.width,workgroups.height,workgroups.depth,
			threads.width,threads.height,threads.depth);

	std::vector<unsigned char> packed(selected.packed_argument_size,0);
	for (std::size_t i = 0; i < argument_count; ++i)
	{
		const auto & source = arguments[i];
		const auto & destination = selected.argument_layouts[i];
		if (source.size != destination.size)
			throw std::runtime_error("Metal kernel argument-size mismatch for " +
				requested_name + " argument " + std::to_string(i) + ": host " +
				std::to_string(source.size) + ", device ABI " +
				std::to_string(destination.size));
		std::memcpy(packed.data() + destination.offset,source.data,source.size);
	}
	if (std::getenv("OPENFPM_METAL_TRACE_ARGUMENTS") != nullptr)
	{
		for (std::size_t i = 0; i < argument_count; ++i)
			std::fprintf(stderr,"  argument %zu: offset=%u size=%u alignment=%u\n",
				i,selected.argument_layouts[i].offset,
				selected.argument_layouts[i].size,
				selected.argument_layouts[i].alignment);
		for (std::size_t offset = 0; offset < packed.size(); offset += 8)
		{
			std::uint64_t word = 0;
			std::memcpy(&word,packed.data() + offset,
				std::min<std::size_t>(sizeof(word),packed.size() - offset));
			std::fprintf(stderr,"  packed[%03zu]=0x%016llx\n",offset,
				static_cast<unsigned long long>(word));
		}
	}

	// The normalized kernel takes one physical-storage-buffer pointer. Its
	// reflected push-constant offset depends on which launch builtins the
	// kernel uses; the packed data itself has no descriptor/std140 ABI.
	MoltenVKMemory argument_buffer(packed.size());
	if (!packed.empty())
	{
		std::memcpy(argument_buffer.getPointer(),packed.data(),packed.size());
		argument_buffer.hostToDevice();
	}
	auto & context = moltenvk_context();
	std::lock_guard<std::recursive_mutex> execution_lock(
		context.execution_mutex());
	VkDevice device = context.device();
	VkCommandBuffer command = VK_NULL_HANDLE;
	auto free_command = [&]
	{
		if (command != VK_NULL_HANDLE)
		{
			vkFreeCommandBuffers(device,context.command_pool(),1,&command);
			command = VK_NULL_HANDLE;
		}
	};
	try
	{
		const std::string cache_key = pipeline_cache_key(selected,threads);
		auto cached = pipeline_cache().find(cache_key);
		if (cached == pipeline_cache().end())
		{
			CachedPipeline created;
			try
			{
				VkShaderModuleCreateInfo module_info{
					VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
				module_info.codeSize = selected.words.size() * sizeof(std::uint32_t);
				module_info.pCode = selected.words.data();
				check(vkCreateShaderModule(device,&module_info,nullptr,&created.module),
					"vkCreateShaderModule");
				VkPushConstantRange module_push{};
				module_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				module_push.size = selected.push_constants.size;
				VkPipelineLayoutCreateInfo layout_info{
					VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
				if (module_push.size != 0)
				{
					layout_info.pushConstantRangeCount = 1;
					layout_info.pPushConstantRanges = &module_push;
				}
				check(vkCreatePipelineLayout(device,&layout_info,nullptr,
					&created.layout),"vkCreatePipelineLayout");
				std::array<std::uint32_t,3> local_size{
					threads.width,threads.height,threads.depth};
				std::array<VkSpecializationMapEntry,3> entries{{
					{0,0,4},{1,4,4},{2,8,4}}};
				VkSpecializationInfo specialization{
					3,entries.data(),sizeof(local_size),local_size.data()};
				VkPipelineShaderStageCreateInfo stage{
					VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
				stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				stage.module = created.module;
				stage.pName = selected.entry.c_str();
				stage.pSpecializationInfo = &specialization;
				VkComputePipelineCreateInfo pipeline_info{
					VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
				pipeline_info.stage = stage;
				pipeline_info.layout = created.layout;
				const VkResult pipeline_result = vkCreateComputePipelines(device,
					VK_NULL_HANDLE,1,&pipeline_info,nullptr,&created.pipeline);
				if (pipeline_result != VK_SUCCESS)
					throw std::runtime_error("vkCreateComputePipelines for " +
						selected.entry + " failed with Vulkan error " +
						std::to_string(static_cast<int>(pipeline_result)));
			}
			catch (...)
			{
				if (created.pipeline != VK_NULL_HANDLE)
					vkDestroyPipeline(device,created.pipeline,nullptr);
				if (created.layout != VK_NULL_HANDLE)
					vkDestroyPipelineLayout(device,created.layout,nullptr);
				if (created.module != VK_NULL_HANDLE)
					vkDestroyShaderModule(device,created.module,nullptr);
				throw;
			}
			cached = pipeline_cache().emplace(cache_key,created).first;
		}
		const VkPipeline pipeline = cached->second.pipeline;
		const VkPipelineLayout pipeline_layout = cached->second.layout;
		VkCommandBufferAllocateInfo allocation{
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		allocation.commandPool = context.command_pool();
		allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocation.commandBufferCount = 1;
		check(vkAllocateCommandBuffers(device,&allocation,&command),
			"vkAllocateCommandBuffers");
		VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		check(vkBeginCommandBuffer(command,&begin),"vkBeginCommandBuffer");
		vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
		std::vector<unsigned char> push_constants(selected.push_constants.size,0);
		auto write_extent = [&](std::int32_t offset, VkExtent3D extent)
		{
			if (offset < 0) return;
			const std::array<std::uint32_t,3> value{
				extent.width,extent.height,extent.depth};
			std::memcpy(push_constants.data() + offset,value.data(),sizeof(value));
		};
		write_extent(selected.push_constants.global_offset,{0,0,0});
		write_extent(selected.push_constants.region_offset,{0,0,0});
		write_extent(selected.push_constants.region_group_offset,{0,0,0});
		write_extent(selected.push_constants.enqueued_local_size,threads);
		write_extent(selected.push_constants.num_workgroups,workgroups);
		VkExtent3D global_size{};
		auto global_component = [](std::uint32_t groups, std::uint32_t local)
		{
			const std::uint64_t value = static_cast<std::uint64_t>(groups) * local;
			if (value > std::numeric_limits<std::uint32_t>::max())
				throw std::overflow_error("Metal global work size exceeds uint32");
			return static_cast<std::uint32_t>(value);
		};
		global_size.width = global_component(workgroups.width,threads.width);
		global_size.height = global_component(workgroups.height,threads.height);
		global_size.depth = global_component(workgroups.depth,threads.depth);
		write_extent(selected.push_constants.global_size,global_size);
		if (selected.push_constants.argument_pointer >= 0)
		{
			const std::uint64_t argument_address =
				argument_buffer.vulkan_device_address();
			std::memcpy(push_constants.data() +
				selected.push_constants.argument_pointer,&argument_address,
				sizeof(argument_address));
		}
		if (!push_constants.empty())
			vkCmdPushConstants(command,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,
				static_cast<std::uint32_t>(push_constants.size()),
				push_constants.data());
		vkCmdDispatch(command,workgroups.width,workgroups.height,workgroups.depth);
		check(vkEndCommandBuffer(command),"vkEndCommandBuffer");
		VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &command;
		check(vkQueueSubmit(context.queue(),1,&submit,VK_NULL_HANDLE),
			"vkQueueSubmit");
		context.synchronize();
		if (std::getenv("OPENFPM_METAL_TRACE_ARGUMENTS") != nullptr)
		{
			for (std::size_t offset = 0; offset + sizeof(std::uint64_t) <= packed.size();
				offset += sizeof(std::uint64_t))
			{
				std::uint64_t address = 0;
				std::memcpy(&address,packed.data() + offset,sizeof(address));
				try
				{
					const auto * mapped = static_cast<const unsigned char *>(
						host_pointer_from_device(reinterpret_cast<const void *>(
							static_cast<std::uintptr_t>(address)),4));
					std::fprintf(stderr,
						"  mapped argument word at %zu -> %p: %02x %02x %02x %02x\n",
						offset,static_cast<const void *>(mapped),mapped[0],mapped[1],
						mapped[2],mapped[3]);
				}
				catch (const std::out_of_range &) {}
			}
		}
	}
	catch (...)
	{
		free_command();
		throw;
	}
	free_command();
}

void destroy_moltenvk_kernel_cache(VkDevice device) noexcept
{
	if (device == VK_NULL_HANDLE) return;
	for (auto & item : pipeline_cache())
	{
		auto & cached = item.second;
		if (cached.pipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(device,cached.pipeline,nullptr);
		if (cached.layout != VK_NULL_HANDLE)
			vkDestroyPipelineLayout(device,cached.layout,nullptr);
		if (cached.module != VK_NULL_HANDLE)
			vkDestroyShaderModule(device,cached.module,nullptr);
	}
	pipeline_cache().clear();
}

void launch_spirv(const std::uint32_t * words, std::size_t word_count,
	const char * entry_point, const void * arguments, std::size_t argument_size,
	VkExtent3D workgroups, VkExtent3D threads)
{
	if (words == nullptr || word_count == 0 || entry_point == nullptr)
		throw std::invalid_argument("invalid MoltenVK kernel module");
	if (argument_size > 4096)
		throw std::invalid_argument("MoltenVK kernel arguments exceed push-constant limit");
	auto & context = moltenvk_context();
	std::lock_guard<std::recursive_mutex> execution_lock(
		context.execution_mutex());
	VkDevice device = context.device();

	VkShaderModule module = VK_NULL_HANDLE;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkCommandBuffer command = VK_NULL_HANDLE;
	try
	{
		VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		module_info.codeSize = word_count * sizeof(std::uint32_t);
		module_info.pCode = words;
		check(vkCreateShaderModule(device, &module_info, nullptr, &module),
			"vkCreateShaderModule");

		VkPushConstantRange push_range{};
		push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		push_range.size = static_cast<std::uint32_t>(argument_size);
		VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		if (argument_size != 0)
		{
			layout_info.pushConstantRangeCount = 1;
			layout_info.pPushConstantRanges = &push_range;
		}
		check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout),
			"vkCreatePipelineLayout");

		std::array<std::uint32_t, 3> local_size{threads.width, threads.height, threads.depth};
		std::array<VkSpecializationMapEntry, 3> entries{{
			{0, 0, sizeof(std::uint32_t)}, {1, 4, sizeof(std::uint32_t)},
			{2, 8, sizeof(std::uint32_t)}}};
		VkSpecializationInfo specialization{};
		specialization.mapEntryCount = static_cast<std::uint32_t>(entries.size());
		specialization.pMapEntries = entries.data();
		specialization.dataSize = sizeof(local_size);
		specialization.pData = local_size.data();
		VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = module;
		stage.pName = entry_point;
		stage.pSpecializationInfo = &specialization;
		VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		pipeline_info.stage = stage;
		pipeline_info.layout = layout;
		check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
			&pipeline), "vkCreateComputePipelines");

		VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		allocation.commandPool = context.command_pool();
		allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocation.commandBufferCount = 1;
		check(vkAllocateCommandBuffers(device, &allocation, &command),
			"vkAllocateCommandBuffers");
		VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		if (argument_size != 0)
			vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<std::uint32_t>(argument_size), arguments);
		vkCmdDispatch(command, workgroups.width, workgroups.height, workgroups.depth);
		check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
		VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &command;
		check(vkQueueSubmit(context.queue(), 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
		context.synchronize();
	}
	catch (...)
	{
		if (command != VK_NULL_HANDLE)
			vkFreeCommandBuffers(device, context.command_pool(), 1, &command);
		if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
		if (layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, layout, nullptr);
		if (module != VK_NULL_HANDLE) vkDestroyShaderModule(device, module, nullptr);
		throw;
	}

	vkFreeCommandBuffers(device, context.command_pool(), 1, &command);
	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, layout, nullptr);
	vkDestroyShaderModule(device, module, nullptr);
}

} // namespace metal
} // namespace openfpm
