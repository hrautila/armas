
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_gvrotate)  && defined(armas_x_gvcompute)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#include "gvrot.h"

/**
 * @brief Compute Givens rotation.
 *
 * Compatible to blas.DROTG and lapack.DLARTG.
 *
 * @ingroup lapack
 */
void armas_x_gvcompute(DTYPE * c, DTYPE * s, DTYPE * r, DTYPE a, DTYPE b)
{
    gvrotg(c, s, r, a, b);
}

/**
 * @brief Apply Givens rotation.
 *
 * Computes
 *```txt
 *     ( v0 )  = G(c, s) * ( y0 )  or ( v0 v1 ) = ( y0 y1 ) * G(c, s)
 *     ( v1 )              ( y1 )
 *
 *     G(c, s) = ( c  s )  => ( v0 ) = ( c*y0 + s*y1 )
 *               (-s  c )     ( v1 )   ( c*y1 - s*y0 )
 *```
 * @ingroup lapack
 */
void armas_x_gvrotate(DTYPE * v0, DTYPE * v1,
                      DTYPE cos, DTYPE sin, DTYPE y0, DTYPE y1)
{
    *v0 = cos * y0 + sin * y1;
    *v1 = cos * y1 - sin * y0;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
