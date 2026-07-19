#ifndef OPENFPM_MOLTENVK_COMPAT_HIPCUB_HPP_
#define OPENFPM_MOLTENVK_COMPAT_HIPCUB_HPP_

// This header is placed ahead of the chipStar include directories only for
// translation units compiled by openfpm_add_moltenvk_kernels().  It preserves
// the normal HIP include used by OpenFPM's core primitive wrappers while
// selecting the deliberately narrow MoltenVK hipCUB facade.
#include "util/metal/MoltenVKPrimitives.hpp"

#endif
