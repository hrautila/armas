
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
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

#if defined(COMPAT)
#if defined(__rot)
void __rot(int *n, DTYPE *X, int *incx, DTYPE *Y, int *incy, DTYPE *c, DTYPE *s)
{
    int i;
    DTYPE x0, y0;
    __armas_dense_t y, x;

    if (*incx == 1) {
        __armas_make(&x, *n, 1, *n, X);
    } else {
        __armas_make(&x, 1, *n, *incx, X);
    }
    if (*incy == 1) {
        __armas_make(&y, *n, 1, *n, Y);
    } else {
        __armas_make(&y, 1, *n, *incy, Y);
    }
    for (i = 0; i < *n; i++) {
        __armas_gvrotate(&x0, &y0, *c, *s,
                         __armas_get_at_unsafe(&x, i), __armas_get_at_unsafe(&y, i));
        __armas_set_at_unsafe(&x, i, x0);
        __armas_set_at_unsafe(&y, i, y0);
    }
}
#endif  // rot

#if defined(__rotg)
void __rotg(DTYPE *sa, DTYPE *sb, DTYPE *c, DTYPE *s)
{
    DTYPE r;
    __armas_gvcompute(c, s, &r, *sa, *sb);
}
#endif // rotg

#if defined(__rotm)
void __rotm(int *n, DTYPE *X, int *incx, DTYPE *Y, int *incy, DTYPE *dparm)
{
    //__armas_dense_t dh;
    //__armas_make(&dh, 2, 2, 2, &dparm[1]);
}
#endif // rotm

#if defined(__rotmg)
// not implemented, yet
void __rotmg(DTYPE *dd1, DTYPE *dd2, DTYPE *dx1, DTYPE *dy1, DTYPE *dparam)
{
    //__armas_dense_t dh;
    //__armas_make(&dh, 2, 2, 2, &dparam[1]);
}
#endif // rotmg
#endif // COMPAT

#if defined(COMPAT_CBLAS) && defined(__cblas_rot)

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
