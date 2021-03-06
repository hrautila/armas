
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(syr2f) || defined(cblas_syr2)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_mvupdate2_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_syr2f)
void blas_syr2f(char *uplo, int *n, DTYPE * alpha, DTYPE * X,
                int *incx, DTYPE * Y, int *incy, DTYPE * A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t a, x, y;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;

    armas_make(&a, *n, *n, *lda, A);
    if (*incx == 1) {
        armas_make(&x, *n, 1, *n, X);
    } else {
        armas_make(&x, 1, *n, *incx, X);
    }
    if (*incy == 1) {
        armas_make(&y, *n, 1, *n, Y);
    } else {
        armas_make(&y, 1, *n, *incy, Y);
    }
    armas_mvupdate2_sym(ONE, &a, *alpha, &x, &y, flags, conf);
}
#endif

#if defined(cblas_syr2)
void cblas_syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                int N, DTYPE alpha, DTYPE * X, int incx, DTYPE * Y, int incy,
                DTYPE * A, int lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t Aa, x, y;
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
    armas_make(&Aa, N, N, lda, A);
    if (incx == 1) {
        armas_make(&x, N, 1, N, X);
    } else {
        armas_make(&x, 1, N, incx, X);
    }
    if (incy == 1) {
        armas_make(&y, N, 1, N, Y);
    } else {
        armas_make(&y, 1, N, incy, Y);
    }
    armas_mvupdate2_sym(ONE, &Aa, alpha, &x, &y, flags, conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
