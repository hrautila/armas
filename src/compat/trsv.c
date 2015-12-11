
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__trsv) || defined(__cblas_trsv)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mvsolve_trm)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__trsv)
void __trsv(char *uplo, char *trans, char *diag, int *n, DTYPE *alpha, DTYPE *A,
            int *lda, DTYPE *X, int *incx)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t a, x;
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
    if (toupper(*trans) == 'T') 
        flags |= ARMAS_TRANS;
    if (toupper(*diag) == 'U') 
        flags |= ARMAS_UNIT;
    
    __armas_make(&a, *n, *n, *lda, A);
    if (*incx == 1) {
        __armas_make(&x, *n, 1, *n, X);
    } else {
        __armas_make(&x, 1, *n, *incx, X);
    }
    __armas_mvsolve_trm(&x, &a, __ONE, flags, conf);
}
#endif

#if defined(__cblas_trsv)
void __cblas_trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, 
                  const enum CBLAS_TRANSPOSE trans,  const enum CBLAS_DIAG diag, int N,
                  DTYPE alpha, DTYPE *A, int lda, DTYPE *X,  int incx)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t Aa, x;
    int flags = 0;

    switch (order) {
    case CblasRowMajor:
        flags |= uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        if (trans == CblasNoTrans)
            flags |= ARMAS_TRANS;
        if (diag == CblasUnit)
            flags |= ARMAS_UNIT;
        break;
    case CblasColMajor:
    default:
        flags |= uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        if (trans == CblasTrans)
            flags |= ARMAS_TRANS;
        if (diag == CblasUnit)
            flags |= ARMAS_UNIT;
        break;
    }
    if (incx == 1) {
        __armas_make(&x, N, 1, N, X);
    } else {
        __armas_make(&x, 1, N, incx, X);
    }
    __armas_make(&Aa, N, N, lda, A);
    __armas_mvsolve_trm(&x, &Aa, alpha, flags, conf);
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
