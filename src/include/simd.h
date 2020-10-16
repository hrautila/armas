
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_SIMD_H
#define __ARMAS_SIMD_H 1

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__x86_64__)
#include "x86_64/simd_ops.h"

#elif defined(__arm__)
#include "arm/simd_ops.h"
#endif

#ifdef __cplusplus
}
#endif

#endif // __SIMD_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
