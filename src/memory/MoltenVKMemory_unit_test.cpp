#include "config.h"

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "memory/MoltenVKMemory.hpp"
#include "util/cuda_util.hpp"
#include "util/metal/MoltenVKContext.hpp"

#include <cstdint>

BOOST_AUTO_TEST_SUITE(MoltenVKMemory_test)

BOOST_AUTO_TEST_CASE(memory_interface_and_device_addresses)
{
	BOOST_REQUIRE_EQUAL(CUDA_ON_BACKEND, CUDA_BACKEND_METAL);
	MoltenVKMemory memory;
	BOOST_REQUIRE(memory.allocate(1024));
	// Match CudaMemory: device storage exists, but memory_c must still construct
	// non-trivial objects placed into it.
	BOOST_REQUIRE(!memory.isInitialized());
	BOOST_REQUIRE_EQUAL(memory.size(), 1024u);
	BOOST_REQUIRE(memory.getPointer() != nullptr);
	BOOST_REQUIRE(memory.getDevicePointer() != nullptr);
	BOOST_REQUIRE_EQUAL(reinterpret_cast<std::uintptr_t>(memory.getDevicePointer()),
		static_cast<std::uintptr_t>(memory.vulkan_device_address()));

	auto * bytes = static_cast<unsigned char *>(memory.getPointer());
	for (std::size_t i = 0; i < memory.size(); ++i)
		bytes[i] = static_cast<unsigned char>(i);
	memory.hostToDevice();
	memory.deviceToHost();
	for (std::size_t i = 0; i < memory.size(); ++i)
		BOOST_REQUIRE_EQUAL(bytes[i], static_cast<unsigned char>(i));

	memory.fill(7);
	memory.deviceToHost();
	for (std::size_t i = 0; i < memory.size(); ++i) BOOST_REQUIRE_EQUAL(bytes[i], 7u);

	// Match CudaMemory's split host/device model even though Apple Silicon uses
	// physically unified memory underneath Vulkan. Neither view becomes current
	// until the corresponding synchronization call is made.
	bytes[0] = 3;
	memory.deviceToHost(0,1);
	BOOST_REQUIRE_EQUAL(bytes[0],7u);
	bytes[0] = 5;
	memory.hostToDevice(0,1);
	bytes[0] = 9;
	memory.deviceToHost(0,1);
	BOOST_REQUIRE_EQUAL(bytes[0],5u);
	memory.fill(7);

	// cudaMemset receives the Vulkan buffer device address, not a CPU pointer.
	// Exercise an interior address so the HIP compatibility runtime must resolve
	// the allocation and offset before publishing the CPU write to the device.
	auto * device_bytes = static_cast<unsigned char *>(memory.getDevicePointer());
	BOOST_REQUIRE_EQUAL(cudaMemset(device_bytes + 128,0xa5,256),cudaSuccess);
	memory.deviceToHost();
	for (std::size_t i = 0; i < memory.size(); ++i)
	{
		const unsigned int expected = i >= 128 && i < 384 ? 0xa5u : 7u;
		BOOST_REQUIRE_EQUAL(bytes[i],expected);
	}
	BOOST_REQUIRE_EQUAL(cudaMemset(device_bytes,0,0),cudaSuccess);

	memory.incRef();
	BOOST_REQUIRE_EQUAL(memory.ref(), 1);
	BOOST_REQUIRE(memory.resize(4096));
	BOOST_REQUIRE_EQUAL(memory.ref(), 1);
	memory.decRef();
	BOOST_REQUIRE_EQUAL(memory.size(), 4096u);
	bytes = static_cast<unsigned char *>(memory.getPointer());
	for (std::size_t i = 0; i < 1024; ++i)
	{
		const unsigned int expected = i >= 128 && i < 384 ? 0xa5u : 7u;
		BOOST_REQUIRE_EQUAL(bytes[i],expected);
	}

	MoltenVKMemory copy(memory.size());
	BOOST_REQUIRE(copy.copy(memory));
	copy.deviceToHost();
	auto * copied = static_cast<const unsigned char *>(copy.getPointer());
	for (std::size_t i = 0; i < 1024; ++i)
	{
		const unsigned int expected = i >= 128 && i < 384 ? 0xa5u : 7u;
		BOOST_REQUIRE_EQUAL(copied[i],expected);
	}

	// CudaMemory-specific fast paths are part of the effective OpenFPM memory
	// contract even though they are not virtual methods on memory. Keep those
	// call sites source-compatible when CudaMemory selects this backend.
	MoltenVKMemory ranged_copy(128);
	ranged_copy.fill(0);
	ranged_copy.deviceToDevice(memory.getDevicePointer(),128,192,32);
	ranged_copy.deviceToHost();
	const auto * ranged =
		static_cast<const unsigned char *>(ranged_copy.getPointer());
	for (std::size_t i = 0; i < ranged_copy.size(); ++i)
		BOOST_REQUIRE_EQUAL(ranged[i],i >= 32 && i < 96 ? 0xa5u : 0u);

	copy.destroy();
	BOOST_REQUIRE(!copy.isInitialized());
	BOOST_REQUIRE_EQUAL(copy.size(), 0u);
}

BOOST_AUTO_TEST_CASE(swap_preserves_cuda_allocator_ownership_contract)
{
	MoltenVKMemory destination(64);
	MoltenVKMemory scratch(128);
	destination.incRef();
	scratch.incRef();
	scratch.incRef();
	const auto destination_address = destination.vulkan_device_address();
	const auto scratch_address = scratch.vulkan_device_address();

	destination.swap(scratch);
	BOOST_REQUIRE_EQUAL(destination.vulkan_device_address(),scratch_address);
	BOOST_REQUIRE_EQUAL(scratch.vulkan_device_address(),destination_address);
	BOOST_REQUIRE_EQUAL(destination.ref(),2);
	// swap_nomode keeps a global scratch allocator externally retained.  This
	// asymmetric detail is the established CudaMemory ownership contract.
	BOOST_REQUIRE_EQUAL(scratch.ref(),2);

	destination.decRef();
	destination.decRef();
	scratch.decRef();
	scratch.decRef();
}

BOOST_AUTO_TEST_SUITE_END()
