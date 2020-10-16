
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_axpyf) || defined(cblas_axpy)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_axpy)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

static
void compat_axpy(const int n, const DTYPE alpha, DTYPE * X, const int incx,
                 DTYPE * Y, const int incy)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t y, x;
    int ix, iy, nx, ny, k;
    DTYPE xv, yv;

    ix = incx < 0 ? -incx : incx;
    iy = incy < 0 ? -incy : incy;

    if (ix == 1) {
        armas_make(&x, n, 1, n, X);
    } else {
        armas_make(&x, 1, n, ix, X);
    }
    if (iy == 1) {
        armas_make(&y, n, 1, n, Y);
    } else {
        armas_make(&y, 1, n, iy, Y);
    }
    if (incx * incy > 0) {
        armas_axpy(&y, alpha, &x, conf);
        return;
    }
    // if not same sign then iteration direction is different (so clever)
    ix = incx < 0 ? n - 1 : 0;
    iy = incy < 0 ? n - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (k = 0; k < n; ix += nx, iy += ny, k++) {
        xv = armas_get_at_unsafe(&x, ix);
        yv = armas_get_at_unsafe(&y, iy);
        armas_set_at_unsafe(&y, iy, yv + alpha * xv);
    }
}

#if defined(blas_axpyf)
void blas_axpyf(int *n, DTYPE * alpha, DTYPE * X, int *incx, DTYPE * Y, int *incy)
{
    compat_axpy(*n, *alpha, X, *incx, Y, *incy);
}
#endif

#if defined(cblas_axpy)
void cblas_axpy(const int N, const DTYPE alpha, DTYPE * X, const int incx,
                  DTYPE * Y, const int incy)
{
    compat_axpy(N, alpha, X, incx, Y, incy);
}
#endif


#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
