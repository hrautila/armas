
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(blas_dotf) || defined(cblas_dot)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_dot)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

static
DTYPE dot_compat(const int N, DTYPE * X, const int incx, DTYPE * Y,
                   const int incy)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t y, x;
    int ix, iy, nx, ny, k;
    DTYPE xv, yv;

    ix = incx < 0 ? -incx : incx;
    iy = incy < 0 ? -incy : incy;

    if (ix == 1) {
        armas_x_make(&x, N, 1, N, X);
    } else {
        armas_x_make(&x, 1, N, ix, X);
    }
    if (iy == 1) {
        armas_x_make(&y, N, 1, N, Y);
    } else {
        armas_x_make(&y, 1, N, iy, Y);
    }
    if (incx * incy > 0) {
        return armas_x_dot(&y, &x, conf);
    }
    DTYPE dval = ZERO;

    // if not same sign then iteration directions are different
    ix = incx < 0 ? N - 1 : 0;
    iy = incy < 0 ? N - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (k = 0; k < N; ix += nx, iy += ny, k++) {
        xv = armas_x_get_at_unsafe(&x, ix);
        yv = armas_x_get_at_unsafe(&y, iy);
        dval += xv * yv;
    }
    return dval;
}

#if defined(blas_dotf)
DTYPE blas_dotf(int *n, DTYPE * X, int *incx, DTYPE * Y, int *incy)
{
    return dot_compat(*n, X, *incx, Y, *incy);
}
#endif


#if defined(cblas_dot)
DTYPE cblas_dot(const int N, DTYPE * X, const int incx, DTYPE * Y,
                  const int incy)
{
    return dot_compat(N, X, incx, Y, incy);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
