
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__swap) || defined(__cblas_swap)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_swap)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) || defined(COMPAT_CBLAS)
static
void __swap_compat(int N, DTYPE *X, int incx, DTYPE *Y, int incy)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t y, x;
    int ix, iy, nx, ny, k;
    DTYPE yv;

    ix = incx < 0 ? -incx : incx;
    iy = incy < 0 ? -incy : incy;

    if (ix == 1) {
        __armas_make(&x, N, 1, N, X);
    } else {
        __armas_make(&x, 1, N, ix, X);
    }
    if (iy == 1) {
        __armas_make(&y, N, 1, N, Y);
    } else {
        __armas_make(&y, 1, N, iy, Y);
    }
    if (incx*incy > 0) {
        __armas_swap(&y, &x, conf);
        return;
    }

    // if not same sign then iteration direction is different (so clever)
    ix = incx < 0 ? N - 1 : 0;
    iy = incy < 0 ? N - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (k = 0; k < N; ix += nx, iy += ny, k++) {
        yv = __armas_get_at_unsafe(&y, iy);
        __armas_set_at_unsafe(&y, iy, __armas_get_at_unsafe(&x, ix));
        __armas_set_at_unsafe(&x, ix, yv);
    }
}
#endif

#if defined(COMPAT) && defined(__swap)
void __swap(int *n, DTYPE *X, int *incx, DTYPE *Y, int *incy)
{
    __swap_compat(*n, X, *incx, Y, *incy);
}
#endif

#if defined(COMPAT_CBLAS) && defined(__cblas_swap)

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
