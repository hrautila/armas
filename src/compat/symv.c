
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__symv) || defined(__cblas_symv)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mvmult_sym)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__symv)
void __symv(char *uplo, int *n, DTYPE *alpha, DTYPE *A,
            int *lda, DTYPE *X, int *incx, DTYPE *beta, DTYPE *Y, int *incy)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t y, a, x;
    int flags = 0;

    switch (toupper(*uplo)) {
    case 'L':
        flags |= ARMAS_LOWER;
        break;
    case 'U':
    default:
        flags |= ARMAS_UPPER;
        break;
    }
    
    __armas_make(&a, *n, *n, *lda, A);
    if (*incy == 1) {
        // column vector
        __armas_make(&y, *n, 1, *n, Y);
    } else {
        // row vector
        __armas_make(&y, 1, *n, *incy, Y);
    }
    if (*incx == 1) {
        __armas_make(&x, *n, 1, *n, X);
    } else {
        __armas_make(&x, 1, *n, *incx, X);
    }
    __armas_mvmult_sym(&y, &a, &x, *alpha, *beta, flags, conf);
}
#endif

#if defined(__cblas_symv)
void __cblas_symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,  
                  int N, DTYPE alpha, DTYPE *A, int lda, 
                  DTYPE *X,  int incx, DTYPE beta, DTYPE *Y, int incy)
{
    __armas_dense_t Aa, x, y;
    armas_conf_t conf = *armas_conf_default();
    int flags;

    switch (order) {
    case CblasRowMajor:
        flags = uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        break;
    case CblasColMajor:
    default:
        flags = uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        break;
    }
    __armas_make(&Aa, N, N, lda, A);
    if (incx == 1) {
        __armas_make(&x, N, 1, N, X);
    } else {
        __armas_make(&x, 1, N, incx, X);
    }
    if (incy == 1) {
        __armas_make(&y, N, 1, N, Y);
    } else {
        __armas_make(&y, 1, N, incy, Y);
    }
    __armas_mvmult_sym(&y, &Aa, &x, alpha, beta, flags, &conf);
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
