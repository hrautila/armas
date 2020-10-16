
// Copyright by libARMAS authors. See AUTHORS file in this archive.

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
#include <float.h>

#ifdef COMPLEX128
/*
 *
 */
#ifdef CONFIG_NOTYPENAMES
#error "NOTYPENAMES config options not working with complex type."
#else
#warn "Complex data types not tested at all"
#endif
#include "complex64/names_base.h"
#include "complex64/names_blas.h"

#elif COMPLEX64
/*
 *
 */
#ifdef CONFIG_NOTYPENAMES
#error "NOTYPENAMES config options not tested with complex type."
#else
#warn "Complex data types not tested at all"
#endif
#include "complex32/names_base.h"
#include "complex32/names_blas.h"

#elif FLOAT32
/*
 *
 */
#include "float32/names_base.h"
#include "float32/names_blas.h"

#else
/*
 *
 */
#include "float64/names_base.h"
#include "float64/names_blas.h"

#endif

#endif /* ARMAS_DTYPE_H */
#endif /* __DOXYGEN */
