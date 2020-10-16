
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_gerf) || defined(cblas_ger)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(ENABLE_COMPAT) && defined(armas_mvupdate)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_gerf)
void blas_gerf(int *m, int *n, DTYPE * alpha, DTYPE * X,
               int *incx, DTYPE * Y, int *incy, DTYPE * A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t y, a, x;

    armas_make(&a, *m, *n, *lda, A);
    if (*incy == 1) {
        // column vector
        armas_make(&y, *n, 1, *n, Y);
    } else {
        // row vector
        armas_make(&y, 1, *n, *incy, Y);
    }
    if (*incx == 1) {
        armas_make(&x, *m, 1, *m, X);
    } else {
        armas_make(&x, 1, *m, *incx, X);
    }
    armas_mvupdate(ONE, &a, *alpha, &x, &y, conf);
}
#endif

#if defined(cblas_ger)
void cblas_ger(const enum CBLAS_ORDER order, const int M,
               const int N, const DTYPE alpha, DTYPE * X, const int incx,
               DTYPE * Y, const int incy, DTYPE * A, const int lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t y, a, x;

    if (incy == 1) {
        // column vector
        armas_make(&y, N, 1, N, Y);
    } else {
        // row vector
        armas_make(&y, 1, N, incy, Y);
    }
    if (incx == 1) {
        armas_make(&x, M, 1, M, X);
    } else {
        armas_make(&x, 1, M, incx, X);
    }

    if (order == CblasRowMajor) {
        armas_make(&a, N, M, lda, A);
        armas_mvupdate(ONE, &a, alpha, &y, &x, conf);
    } else {
        armas_make(&a, M, N, lda, A);
        armas_mvupdate(ONE, &a, alpha, &x, &y, conf);
    }
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
