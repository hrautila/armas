
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_swapf) || defined(cblas_swap)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_swap)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

static
void swap_compat(const int N, DTYPE * X, const int incx, DTYPE * Y,
                   const int incy)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t y, x;
    int ix, iy, nx, ny, k;
    DTYPE yv;

    ix = incx < 0 ? -incx : incx;
    iy = incy < 0 ? -incy : incy;

    if (ix == 1) {
        armas_make(&x, N, 1, N, X);
    } else {
        armas_make(&x, 1, N, ix, X);
    }
    if (iy == 1) {
        armas_make(&y, N, 1, N, Y);
    } else {
        armas_make(&y, 1, N, iy, Y);
    }
    if (incx * incy > 0) {
        armas_swap(&y, &x, conf);
        return;
    }
    // if not same sign then iteration direction is different (so clever)
    ix = incx < 0 ? N - 1 : 0;
    iy = incy < 0 ? N - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (k = 0; k < N; ix += nx, iy += ny, k++) {
        yv = armas_get_at_unsafe(&y, iy);
        armas_set_at_unsafe(&y, iy, armas_get_at_unsafe(&x, ix));
        armas_set_at_unsafe(&x, ix, yv);
    }
}

#if defined(blas_swapf)
void blas_swapf(int *n, DTYPE * X, int *incx, DTYPE * Y, int *incy)
{
    swap_compat(*n, X, *incx, Y, *incy);
}
#endif

#if defined(cblas_swap)
void cblas_swap(const int N, DTYPE * X, const int incx, DTYPE * Y,
                  const int incy)
{
    swap_compat(N, X, incx, Y, incy);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
