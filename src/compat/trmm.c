
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(trmmf) || defined(cblas_trmm)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_trm)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_trmmf)
void blas_trmmf(char *side, char *uplo, char *transa, char *diag, int *m,
                int *n, DTYPE * alpha, DTYPE * A, int *lda, DTYPE * B, int *ldb)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t a, b;
    int flags = 0;

    armas_x_make(&b, *m, *n, *ldb, B);

    switch (toupper(*side)) {
    case 'R':
        flags |= ARMAS_RIGHT;
        armas_x_make(&a, *n, *n, *lda, A);
        break;
    case 'L':
    default:
        flags |= ARMAS_LEFT;
        armas_x_make(&a, *m, *m, *lda, A);
        break;
    }
    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;
    if (toupper(*transa) == 'T')
        flags |= ARMAS_TRANS;
    if (toupper(*diag) == 'U')
        flags |= ARMAS_UNIT;

    armas_x_mult_trm(&b, *alpha, &a, flags, conf);
}
#endif

#if defined(cblas_trmm)
void cblas_trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa,
                const enum CBLAS_DIAG diag, int M, int N,
                DTYPE alpha, DTYPE * A, int lda, DTYPE * B, int ldb)
{
    armas_conf_t conf = *armas_conf_default();
    armas_x_dense_t Aa, Ba;
    int flags = 0;
    switch (order) {
    case CblasColMajor:
        flags |= side == CblasRight ? ARMAS_RIGHT : ARMAS_LEFT;
        flags |= uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        if (diag == CblasUnit)
            flags |= ARMAS_UNIT;
        if (transa == CblasTrans)
            flags |= ARMAS_TRANSA;
        // M > ldb --> error
        armas_x_make(&Ba, M, N, ldb, B);
        if (side == CblasRight) {
            // N > lda --> error
            armas_x_make(&Aa, N, N, lda, A);
        } else {
            // M > lda --> error
            armas_x_make(&Aa, M, M, lda, A);
        }
        break;
    case CblasRowMajor:
        flags |= side == CblasRight ? ARMAS_LEFT : ARMAS_RIGHT;
        flags |= uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        if (diag == CblasUnit)
            flags |= ARMAS_UNIT;
        if (transa == CblasNoTrans)
            flags |= ARMAS_TRANSA;
        // N > ldb --> error
        armas_x_make(&Ba, N, M, ldb, B);
        if (side == CblasRight) {
            // N > lda --> error
            armas_x_make(&Aa, M, M, lda, A);
        } else {
            // M > lda --> error
            armas_x_make(&Aa, N, N, lda, A);
        }
        break;
    default:
        return;
    }
    armas_x_mult_trm(&Ba, alpha, &Aa, flags, &conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
