
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_trsvf) || defined(cblas_trsv)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_mvsolve_trm)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_trsvf)
void blas_trsvf(char *uplo, char *trans, char *diag, int *n, DTYPE * A,
                int *lda, DTYPE * X, int *incx)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t a, x;
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

    armas_make(&a, *n, *n, *lda, A);
    if (*incx == 1) {
        armas_make(&x, *n, 1, *n, X);
    } else {
        armas_make(&x, 1, *n, *incx, X);
    }
    armas_mvsolve_trm(&x, ONE, &a, flags, conf);
}
#endif

#if defined(cblas_trsv)
void cblas_trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                int N, DTYPE alpha, DTYPE * A, int lda, DTYPE * X, int incx)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t Aa, x;
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
        armas_make(&x, N, 1, N, X);
    } else {
        armas_make(&x, 1, N, incx, X);
    }
    armas_make(&Aa, N, N, lda, A);
    armas_mvsolve_trm(&x, alpha, &Aa, flags, conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
