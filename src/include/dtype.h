
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN
#ifndef ARMAS_DTYPE_H
#define ARMAS_DTYPE_H 1

#if HAVE_CONFIG_H
  #include "config.h"
#endif

#include <math.h>

#ifdef COMPLEX128
/*
 *
 */
#elif COMPLEX64
/*
 *
 */

#elif FLOAT32
/*
 *
 */

#else
/*
 *
 */
#include "float64/names_base.h"
#include "float64/names_blas.h"

#endif

#endif /* ARMAS_DTYPE_H */
#endif /* __DOXYGEN */