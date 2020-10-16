
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"
// givens functions are here
#include "dlpack.h"

// ----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_rotf) || defined(blas_rotgf) || defined(cblas_rot)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_gvrotate) && defined(armas_gvcompute)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"


static
void rot_compat(int N, DTYPE * X, int incx, DTYPE * Y, int incy, DTYPE c,
                DTYPE s)
{
    armas_dense_t y, x;
    int ix, iy, nx, ny, i;
    DTYPE x0, y0;

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

    ix = incx < 0 ? N - 1 : 0;
    iy = incy < 0 ? N - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (i = 0; i < N; i++, iy += ny, ix += nx) {
        armas_gvrotate(&x0, &y0, c, s,
                         armas_get_at_unsafe(&x, ix),
                         armas_get_at_unsafe(&y, iy));
        armas_set_at_unsafe(&x, ix, x0);
        armas_set_at_unsafe(&y, iy, y0);
    }
}

#if defined(blas_rotf)
void blas_rotf(int *n, DTYPE * X, int *incx, DTYPE * Y, int *incy, DTYPE * c,
            DTYPE * s)
{
    rot_compat(*n, X, *incx, Y, *incy, *c, *s);
}
#endif

#if defined(blas_rotgf)
void blas_rotgf(DTYPE * sa, DTYPE * sb, DTYPE * c, DTYPE * s)
{
    DTYPE r;
    armas_gvcompute(c, s, &r, *sa, *sb);
}
#endif

#if defined(cblas_rot)

#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
