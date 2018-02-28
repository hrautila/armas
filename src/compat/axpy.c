
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__axpyf) || defined(__cblas_axpy)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_axpy)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

static
void __compat_axpy(const int n, const DTYPE alpha, DTYPE *X, const int incx, DTYPE *Y, const int incy)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t y, x;
    int ix, iy, nx, ny, k;
    DTYPE xv, yv;

    ix = incx < 0 ? - incx : incx;
    iy = incy < 0 ? - incy : incy;

    if (ix == 1) {
        armas_x_make(&x, n, 1, n, X);
    } else {
        armas_x_make(&x, 1, n, ix, X);
    }
    if (iy == 1) {
        armas_x_make(&y, n, 1, n, Y);
    } else {
        armas_x_make(&y, 1, n, iy, Y);
    }
    if (incx*incy > 0) {
        armas_x_axpy(&y, alpha, &x, conf);
        return;
    } 
    // if not same sign then iteration direction is different (so clever)
    ix = incx < 0 ? n - 1 : 0;
    iy = incy < 0 ? n - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (k = 0; k < n; ix += nx, iy += ny, k++) {
        xv = armas_x_get_at_unsafe(&x, ix);
        yv = armas_x_get_at_unsafe(&y, iy);
        armas_x_set_at_unsafe(&y, iy, yv + alpha*xv);
    }
}

#if defined(__axpyf)
void __axpyf(int *n, DTYPE *alpha, DTYPE *X, int *incx, DTYPE *Y, int *incy)
{
    __compat_axpy(*n, *alpha, X, *incx, Y, *incy);
}
#endif

#if defined(__cblas_axpy)
void __cblas_axpy(const int N, const DTYPE alpha, DTYPE *X, const int incx, DTYPE *Y, const int incy)
{
    __compat_axpy(N, alpha, X, incx, Y, incy);
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
