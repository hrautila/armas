
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#define ARMAS_PROVIDES 1
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include "matrix.h"

#ifdef __ARMAS_INLINE
#undef __ARMAS_INLINE
#endif

#define __ARMAS_INLINE
#include "sparse.h"

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
