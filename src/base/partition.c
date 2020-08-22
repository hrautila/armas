
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "matrix.h"
#include "internal.h"

// undefined __ARMAS_INLINE macro 
#ifdef __ARMAS_INLINE
#undef __ARMAS_INLINE
#endif
// define __ARMAS_INLINE as empty 
#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE
#endif

#include "partition.h"
