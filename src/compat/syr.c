
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/brautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(blas_syrf) || defined(cblas_syr)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mvupdate_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_syrf)
void blas_syrf(char *uplo, int *n, DTYPE * alpha, DTYPE * X,
               int *incx, DTYPE * A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t a, x;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;

    armas_x_make(&a, *n, *n, *lda, A);
    if (*incx == 1) {
        armas_x_make(&x, *n, 1, *n, X);
    } else {
        armas_x_make(&x, 1, *n, *incx, X);
    }
    armas_x_mvupdate_sym(ONE, &a, *alpha, &x, flags, conf);
}
#endif

#if defined(cblas_syr)
void cblas_syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
               const int N, const DTYPE alpha, DTYPE * X, const int incx,
               DTYPE * A, const int lda)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t Aa, x;
    int flags = 0;

    if (order == CblasRowMajor) {
        flags = uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
    } else {
        flags = uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
    }

    armas_x_make(&Aa, N, N, lda, A);
    if (incx == 1) {
        armas_x_make(&x, N, 1, N, X);
    } else {
        armas_x_make(&x, 1, N, incx, X);
    }
    armas_x_mvupdate_sym(ONE, &Aa, alpha, &x, flags, conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
