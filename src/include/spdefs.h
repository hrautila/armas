
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN
#ifndef ARMAS_SPDEFS_H
#define ARMAS_SPDEFS_H 1

#if HAVE_CONFIG_H
  #include "config.h"
#endif

#include <math.h>

#ifdef COMPLEX128
/*
 *
 */
#include "complex64/names_base.h"
#include "complex64/names_blas.h"

#elif COMPLEX64
/*
 *
 */
#include "complex32/names_base.h"
#include "complex32/names_blas.h"

#elif FLOAT32
/*
 *
 */
#include "float32/names_base.h"
#include "float32/names_blas.h"
#include "float32/names_sparese.h"

#else
/*
 *
 */
#include "float64/names_base.h"
#include "float64/names_blas.h"
#include "float64/names_sparse.h"

#endif

#endif /* ARMAS_DTYPE_H */
#endif /* __DOXYGEN */
