
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "internal.h"

// include inline functions without extern inline declaration
#ifdef __INLINE
#undef __INLINE
#endif
#define __INLINE
#include "matrix.h"

