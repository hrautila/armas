
// Copyright (c) Harri Rautila, 2014-2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"
// givens functions are here
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__rot) || defined(__rotg) || defined(__cblas_rot)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_gvrotate) && defined(__armas_gvcompute)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"


static
void __rot_compat(int N, DTYPE *X, int incx, DTYPE *Y, int incy, DTYPE c, DTYPE s)
{
    __armas_dense_t y, x;
    int ix, iy, nx, ny, i;
    DTYPE x0, y0;

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

    ix = incx < 0 ? N - 1 : 0;
    iy = incy < 0 ? N - 1 : 0;
    nx = ix == 0 ? 1 : -1;
    ny = iy == 0 ? 1 : -1;
    for (i = 0; i < N; i++, iy += ny, ix += nx) {
        __armas_gvrotate(&x0, &y0, c, s,
                         __armas_get_at_unsafe(&x, ix), __armas_get_at_unsafe(&y, iy));
        __armas_set_at_unsafe(&x, ix, x0);
        __armas_set_at_unsafe(&y, iy, y0);
    }
}

#if defined(__rot)
void __rot(int *n, DTYPE *X, int *incx, DTYPE *Y, int *incy, DTYPE *c, DTYPE *s)
{
    __rot_compat(*n, X, *incx, Y, *incy, *c, *s);
}
#endif

#if defined(__rotg)
void __rotg(DTYPE *sa, DTYPE *sb, DTYPE *c, DTYPE *s)
{
    DTYPE r;
    __armas_gvcompute(c, s, &r, *sa, *sb);
}
#endif // rotg

#if defined(COMPAT_CBLAS) && defined(__cblas_rot)

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
