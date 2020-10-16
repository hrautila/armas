
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN
#ifndef ARMAS_DLPACK_H
#define ARMAS_DLPACK_H 1

#if HAVE_CONFIG_H
  #include "config.h"
#endif

#include <math.h>

#ifdef COMPLEX64
/*
 *
 */
#include "complex64/names_base.h"
#include "complex64/names_blas.h"
#include "complex64/names_lapack.h"

#elif COMPLEX32
/*
 *
 */
#include "complex32/names_base.h"
#include "complex32/names_blas.h"
#include "complex32/names_lapack.h"

#elif FLOAT32
/*
 *
 */
#include "float32/names_base.h"
#include "float32/names_blas.h"
#include "float32/names_lapack.h"

#else
/*
 *
 */
#include "float64/names_base.h"
#include "float64/names_blas.h"
#include "float64/names_lapack.h"

#endif

#endif /* ARMAS_DLPACK_H */
#endif /* __DOXYGEN */
