
// Copyright (c) Harri Rautila, 2014-2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__gemv) || defined(__cblas_gemv)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_gemv)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__gemv)
void __gemv(char *trans, int *m, int *n, DTYPE *alpha, DTYPE *A,
            int *lda, DTYPE *X, int *incx, DTYPE *beta, DTYPE *Y, int *incy)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t y, a, x;
    int ylen, xlen, flags = 0;

    if (toupper(*trans) == 'T') 
        flags |= ARMAS_TRANS;
    
    ylen = flags & ARMAS_TRANS ? *n : *m;
    xlen = flags & ARMAS_TRANS ? *m : *n;
    __armas_make(&a, *m, *n, *lda, A);
    if (*incy == 1) {
        // column vector
        __armas_make(&y, ylen, 1, ylen, Y);
    } else {
        // row vector
        __armas_make(&y, 1, ylen, *incy, Y);
    }
    if (*incx == 1) {
        __armas_make(&x, xlen, 1, xlen, X);
    } else {
        __armas_make(&x, 1, xlen, *incx, X);
    }
    __armas_mvmult(&y, &a, &x, *alpha, *beta, flags, conf);
}
#endif

#if defined(__cblas_gemv)
void __cblas_gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANS trans,  
                  const int M, const int N, const DTYPE alpha, DTYPE *A, const int lda, 
                  DTYPE *X,  const int incx, const DTYPE beta, DTYPE *Y, const int incy)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t y, Aa, x;
    int ylen, xlen, flags = 0;

    switch (order) {
    case CblasRowMajor:
        if (trans == CblasNoTrans)
            flags |= ARMAS_TRANS;
        ylen = flags & ARMAS_TRANS ? M : N;
        xlen = flags & ARMAS_TRANS ? N : M;
        __armas_make(&Aa, N, M, lda, A);
        break;
    case CblasColMajor:
    default:
        if (trans == CblasTrans)
            flags |= ARMAS_TRANS;
        ylen = flags & ARMAS_TRANS ? N : M;
        xlen = flags & ARMAS_TRANS ? M : N;
        __armas_make(&Aa, M, N, lda, A);
        break;
    }

    if (incy == 1) {
        // column vector
        __armas_make(&y, ylen, 1, ylen, Y);
    } else {
        // row vector
        __armas_make(&y, 1, ylen, incy, Y);
    }
    if (incx == 1) {
        __armas_make(&x, xlen, 1, xlen, X);
    } else {
        __armas_make(&x, 1, xlen, incx, X);
    }
    __armas_mvmult(&y, &Aa, &x, alpha, beta, flags, conf);
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
