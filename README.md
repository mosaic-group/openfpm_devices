# openfpm_devices

## Metal backend status

The `METAL` backend is experimental and targets Apple Silicon through Vulkan
and MoltenVK. `MoltenVKMemory` implements the existing OpenFPM memory interface
with host-visible Vulkan buffers and buffer device addresses. The existing
`CudaMemory` and `vector_dist_gpu` APIs select that implementation in Metal
builds; there is no Metal-specific container API.

`openfpm_add_moltenvk_kernels()` compiles the device half of a HIP/CUDA-style
translation unit, lowers it with clspv, embeds the resulting SPIR-V in generated
C++, and registers all discovered kernel entries at startup. The same source is
compiled for the host half by HIP Clang, so the compiler-emitted kernel handle
selects the exact template specialization at launch. `CUDA_LAUNCH`,
`CUDA_LAUNCH_DIM3`, and the lambda/TLS launch wrappers therefore keep their
existing source spelling in translated `.cu` files. A host-only C++ translation
unit cannot contain a device lambda launch and rejects it at compile time; add
that source to `openfpm_add_moltenvk_kernels()` instead.

Kernel arguments use one explicit physical-buffer ABI. A build-time LLVM pass
packs the original parameter list and reconstructs aggregate arguments at the
byte offsets from Clang's `DataLayout`. This is important for nested OpenFPM
views: clspv's aggregate layout is not allowed to remove C++ ABI padding. The
same pass normalizes physical-buffer atomics and the valid SPIR-V forms which
MoltenVK 1.4.1's bundled SPIRV-Cross cannot otherwise lower correctly.

The backend also owns a build-time translated primitive module. Existing
`openfpm::scan`, `reduce`, `segreduce`, `sort`, and `merge` calls dispatch to
device-resident SPIR-V kernels for device-supported scalar types; no
mapped-memory or STL fallback is used for those paths.
Current device typed coverage is `int32`, `uint32`, `int64`, `uint64`, and
Float for scan, reduction, and segmented reduction, and for stable sort and
merge keys. Values paired with sort/merge keys may be any trivially-copyable
type. Apple GPUs have no native FP64: double reduction and segmented reduction
use an explicitly synchronized shared-buffer CPU compatibility path, while
double scan, sort, and merge fail as unsupported.

Verified launches include the existing `Vector/1_gpu_first_step` example and
normal `vector_dist_gpu`, grid, sparse-grid, cell-list, block-map, and AMR
translation units using unchanged launch syntax. Metal builds translate the
existing non-performance GPU test translation units from `openfpm_data`,
`openfpm_pdata`, and `openfpm_vcluster`; no separate Metal algorithm catalogue
is maintained.
Those sources cover the existing scan, reduction, segmented-reduction,
stable-sort, merge, scatter/gather, stencil, ghost exchange, decomposition, and
AMR call sites. As with CUDA and HIP, this finite test catalogue cannot
instantiate every possible user template combination.

MPI transport is not GPU-direct on this backend: the queue is synchronized and
MoltenVK's coherent, persistently mapped device allocation is exposed to MPI at
the transport boundary, while the public `RUN_ON_DEVICE` API and the existing
packing kernels remain unchanged.

Dynamic shared-memory launch bytes are not supported yet. CUDA-only binary
libraries and arbitrary CUDA runtime APIs are outside this translation layer;
ordinary OpenFPM HIP/CUDA kernels using the supported cudify surface are its
target.

Configure with:

```sh
cmake -S . -B build-metal -DCUDA_ON_BACKEND=METAL \
  -DOPENFPM_MOLTENVK_HIP_CLANG=/path/to/clang++ \
  -DOPENFPM_MOLTENVK_LLVM_OPT=/path/to/opt \
  -DOPENFPM_MOLTENVK_CLSPV=/path/to/clspv \
  -DOPENFPM_CHIPSTAR_ROOT=/path/to/chipStar
cmake --build build-metal -j
ctest --test-dir build-metal --output-on-failure
```

The HIP/SPIR-V compiler, matching `llvm-opt`, clspv, and chipStar headers are
required for a Metal build because the core primitive module is generated and
embedded by `ofpmmemory`. Vulkan-Headers and MoltenVK provide the runtime.
