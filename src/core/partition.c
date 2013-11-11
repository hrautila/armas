
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "internal.h"
#include "matrix.h"

// undefined __INLINE macro 
#ifdef __INLINE
#undef __INLINE
#endif
// define __INLINE as empty 
#ifndef __INLINE
#define __INLINE
#endif

#include "partition.h"

