
// Copyright (c) Harri Rautila, 2014-2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__gerf) || defined(__cblas_ger)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(ENABLE_COMPAT) && defined(armas_x_mvupdate)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__gerf)
void __gerf(int *m, int *n, DTYPE *alpha, DTYPE *X,
            int *incx, DTYPE *Y, int *incy, DTYPE *A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t y, a, x;

    armas_x_make(&a, *m, *n, *lda, A);
    if (*incy == 1) {
        // column vector
        armas_x_make(&y, *n, 1, *n, Y);
    } else {
        // row vector
        armas_x_make(&y, 1, *n, *incy, Y);
    }
    if (*incx == 1) {
        armas_x_make(&x, *m, 1, *m, X);
    } else {
        armas_x_make(&x, 1, *m, *incx, X);
    }
    armas_x_mvupdate(&a, &x, &y, *alpha, conf);
}
#endif

#if defined(__cblas_ger)
void __cblas_ger(const enum CBLAS_ORDER order, const int M,  
                 const int N, const DTYPE alpha, DTYPE *X, const int incx, 
                 DTYPE *Y, const int incy, DTYPE *A, const int lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t y, a, x;

    if (incy == 1) {
        // column vector
        armas_x_make(&y, N, 1, N, Y);
    } else {
        // row vector
        armas_x_make(&y, 1, N, incy, Y);
    }
    if (incx == 1) {
        armas_x_make(&x, M, 1, M, X);
    } else {
        armas_x_make(&x, 1, M, incx, X);
    }

    if (order == CblasRowMajor) {
        armas_x_make(&a, N, M, lda, A);
        armas_x_mvupdate(&a, &y, &x, alpha, conf);
    } else {
        armas_x_make(&a, M, N, lda, A);
        armas_x_mvupdate(&a, &x, &y, alpha, conf);
    }
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
