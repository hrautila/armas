
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__syr2f) || defined(__cblas_syr2)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mvupdate2_sym)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__syr2f)
void __syr2f(char *uplo, int *n, DTYPE *alpha, DTYPE *X,
             int *incx, DTYPE *Y, int *incy, DTYPE *A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t a, x, y;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;

    armas_x_make(&a, *n, *n, *lda, A);
    if (*incx == 1) {
        armas_x_make(&x, *n, 1, *n, X);
    } else {
        armas_x_make(&x, 1, *n, *incx, X);
    }
    if (*incy == 1) {
        armas_x_make(&y, *n, 1, *n, Y);
    } else {
        armas_x_make(&y, 1, *n, *incy, Y);
    }
    armas_x_mvupdate2_sym(&a, *alpha, &x, &y, flags, conf);
}
#endif

#if defined(__cblas_syr2)
void __cblas_syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, int N,
                  DTYPE alpha, DTYPE *X, int incx, DTYPE *Y, int incy, DTYPE *A, int lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t Aa, x, y;
    int flags = 0;

    switch (order) {
    case CblasRowMajor:
        flags |= uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        break;
    case CblasColMajor:
    default:
        flags |= uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        break;
    }
    armas_x_make(&Aa, N, N, lda, A);
    if (incx == 1) {
        armas_x_make(&x, N, 1, N, X);
    } else {
        armas_x_make(&x, 1, N, incx, X);
    }
    if (incy == 1) {
        armas_x_make(&y, N, 1, N, Y);
    } else {
        armas_x_make(&y, 1, N, incy, Y);
    }
    armas_x_mvupdate2_sym(&Aa, alpha, &x, &y, flags, conf);
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
