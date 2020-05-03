
// Copyright (c) Harri Rautila, 2015-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(blas_iamaxf) || defined(cblas_iamax)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_iamax)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_iamaxf)
int blas_iamaxf(int *n, DTYPE * X, int *incx)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t x;

    if (*incx == 1) {
        armas_x_make(&x, *n, 1, *n, X);
    } else {
        armas_x_make(&x, 1, *n, *incx, X);
    }
    return armas_x_iamax(&x, conf);
}
#endif

#if defined(cblas_iamax)
int cblas_iamax(int N, DTYPE * X, int incx)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t x;

    if (incx == 1) {
        armas_x_make(&x, N, 1, N, X);
    } else {
        armas_x_make(&x, 1, N, incx, X);
    }
    return armas_x_iamax(&x, conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
