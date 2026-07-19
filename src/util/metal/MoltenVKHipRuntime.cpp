#include "util/cuda_util.hpp"
#include "memory/MoltenVKMemory.hpp"
#include "util/metal/MoltenVKKernel.hpp"

#include <cstring>
#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ihipEvent_t
{
	std::chrono::steady_clock::time_point timestamp;
	bool recorded = false;
};

namespace
{
struct HipLaunchConfiguration
{
	dim3 grid;
	dim3 block;
	std::size_t shared_memory;
	void * stream;
};

thread_local std::vector<HipLaunchConfiguration> launch_configurations;
thread_local std::string last_launch_error;
}

// hipLaunchKernel is a C ABI entry point and therefore reports failures through
// its return value.  hipLaunchKernelGGL hides that value inside its generated
// launch stub, however, and the historical CUDA_LAUNCH call sites do not make
// a second HIP error query.  The Metal launch macros call this immediately
// after the unchanged HIP launch expression so pipeline/dispatch failures are
// observable by ordinary C++ tests and applications.
extern "C" void openfpmMetalCheckLastLaunch()
{
	if (last_launch_error.empty()) return;
	std::string message = std::move(last_launch_error);
	last_launch_error.clear();
	throw std::runtime_error(message);
}

extern "C" void ** __hipRegisterFatBinary(const void * data)
{
	// The SPIR-V and its compiler-generated ABI are registered by the embedded
	// C++ registrar.  HIP clang still requires a non-null module token while
	// registering exact host function handles.
	return reinterpret_cast<void **>(const_cast<void *>(data));
}

extern "C" void __hipUnregisterFatBinary(void *) {}

extern "C" void __hipRegisterFunction(void **, const void * host_function,
	char *, const char * device_name, unsigned int, void *, void *, dim3 *,
	dim3 *, int *)
{
	openfpm::metal::register_kernel_handle(host_function,device_name);
}

extern "C" void __hipRegisterVar(void **, void *, char *, char *, int, int,
	int, int)
{}

extern "C" int __hipPushCallConfiguration(dim3 grid, dim3 block,
	std::size_t shared_memory, void * stream)
{
	launch_configurations.push_back({grid,block,shared_memory,stream});
	return 0;
}

extern "C" int __hipPopCallConfiguration(dim3 * grid, dim3 * block,
	std::size_t * shared_memory, void ** stream)
{
	if (launch_configurations.empty()) return 1;
	const auto configuration = launch_configurations.back();
	launch_configurations.pop_back();
	*grid = configuration.grid;
	*block = configuration.block;
	*shared_memory = configuration.shared_memory;
	*stream = configuration.stream;
	return 0;
}

extern "C" int hipLaunchKernel(const void * host_function, dim3 grid,
	dim3 block, void ** arguments, std::size_t shared_memory, void *)
{
	last_launch_error.clear();
	if (shared_memory != 0)
	{
		last_launch_error =
			"Metal HIP bridge does not support dynamic shared memory";
		std::fprintf(stderr,"Metal HIP kernel launch failed: %s\n",
			last_launch_error.c_str());
		return 1;
	}
	try
	{
		openfpm::metal::launch_registered_handle(host_function,arguments,
			openfpm::metal::to_extent(grid),openfpm::metal::to_extent(block));
		return 0;
	}
	catch (const std::exception & error)
	{
		last_launch_error = error.what();
		std::fprintf(stderr,"Metal HIP kernel launch failed: %s\n",error.what());
		return 1;
	}
}

extern "C" int hipDeviceSynchronize()
{
	try
	{
		openfpm::metal::metal_synchronize();
		return 0;
	}
	catch (const std::exception & error)
	{
		std::fprintf(stderr,"Metal HIP synchronization failed: %s\n",error.what());
		return 1;
	}
}

extern "C" int hipEventCreate(ihipEvent_t ** event)
{
	if (event == nullptr) return 1;
	try
	{
		*event = new ihipEvent_t;
		return 0;
	}
	catch (...)
	{
		*event = nullptr;
		return 1;
	}
}

extern "C" int hipEventDestroy(ihipEvent_t * event)
{
	delete event;
	return 0;
}

extern "C" int hipEventRecord(ihipEvent_t * event, void *)
{
	if (event == nullptr) return 1;
	try
	{
		openfpm::metal::metal_synchronize();
		event->timestamp = std::chrono::steady_clock::now();
		event->recorded = true;
		return 0;
	}
	catch (const std::exception & error)
	{
		std::fprintf(stderr,"Metal HIP event record failed: %s\n",error.what());
		return 1;
	}
}

extern "C" int hipEventSynchronize(ihipEvent_t * event)
{
	if (event == nullptr) return 1;
	return hipDeviceSynchronize();
}

extern "C" int hipEventElapsedTime(float * milliseconds,
	ihipEvent_t * start, ihipEvent_t * stop)
{
	if (milliseconds == nullptr || start == nullptr || stop == nullptr ||
		!start->recorded || !stop->recorded)
		return 1;
	const auto elapsed = std::chrono::duration<double,std::milli>(
		stop->timestamp - start->timestamp);
	*milliseconds = static_cast<float>(elapsed.count());
	return 0;
}

extern "C" int hipMemcpy(void * destination, const void * source,
	std::size_t bytes, int kind)
{
	try
	{
		if (bytes == 0) return 0;

		auto device_mapping = [bytes](const void * pointer, bool & is_device)
			-> void *
		{
			try
			{
				void * mapped = openfpm::metal::host_pointer_from_device(pointer,bytes);
				is_device = true;
				return mapped;
			}
			catch (const std::out_of_range &)
			{
				is_device = false;
				return const_cast<void *>(pointer);
			}
		};

		bool destination_is_device = false;
		bool source_is_device = false;
		void * mapped_destination = destination;
		void * mapped_source = const_cast<void *>(source);
		if (kind == 1 || kind == 3 || kind == 1024)
		{
			mapped_destination = openfpm::metal::host_pointer_from_device(destination,bytes);
			destination_is_device = true;
		}
		if (kind == 2 || kind == 3 || kind == 1024)
		{
			mapped_source = openfpm::metal::host_pointer_from_device(source,bytes);
			source_is_device = true;
		}
		if (kind == 4)
		{
			mapped_destination = device_mapping(destination,destination_is_device);
			mapped_source = device_mapping(source,source_is_device);
		}
		if (kind < 0 || (kind > 4 && kind != 1024)) return 1;

		if (destination_is_device || source_is_device)
			openfpm::metal::metal_synchronize();
		std::memmove(mapped_destination,mapped_source,bytes);
		if (destination_is_device)
			openfpm::metal::flush_device_pointer(destination,bytes);
		return 0;
	}
	catch (const std::exception & error)
	{
		std::fprintf(stderr,"Metal HIP memcpy failed: %s\n",error.what());
		return 1;
	}
}

extern "C" int hipMemset(void * destination, int value, std::size_t bytes)
{
	try
	{
		if (bytes == 0) return 0;
		openfpm::metal::metal_synchronize();
		void * mapped_destination =
			openfpm::metal::host_pointer_from_device(destination,bytes);
		std::memset(mapped_destination,value,bytes);
		openfpm::metal::flush_device_pointer(destination,bytes);
		return 0;
	}
	catch (const std::exception & error)
	{
		std::fprintf(stderr,"Metal HIP memset failed: %s\n",error.what());
		return 1;
	}
}
