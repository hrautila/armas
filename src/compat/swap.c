
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__swapf) || defined(__cblas_swap)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_swap)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

static
void __swap_compat(const int N, DTYPE *X, const int incx, DTYPE *Y, const int incy)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t y, x;
    int ix, iy, nx, ny, k;
    DTYPE yv;

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
    if (incx*incy > 0) {
        armas_x_swap(&y, &x, conf);
        return;
    }

    // if not same sign then iteration direction is different (so clever)
    ix = incx < 0 ? N - 1 : 0;
    iy = incy < 0 ? N - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (k = 0; k < N; ix += nx, iy += ny, k++) {
        yv = armas_x_get_at_unsafe(&y, iy);
        armas_x_set_at_unsafe(&y, iy, armas_x_get_at_unsafe(&x, ix));
        armas_x_set_at_unsafe(&x, ix, yv);
    }
}

#if defined(__swapf)
void __swapf(int *n, DTYPE *X, int *incx, DTYPE *Y, int *incy)
{
    __swap_compat(*n, X, *incx, Y, *incy);
}
#endif

#if defined(__cblas_swap)
void __cblas_swap(const int N, DTYPE *X, const int incx, DTYPE *Y, const int incy)
{
    __swap_compat(N, X, incx, Y, incy);
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
