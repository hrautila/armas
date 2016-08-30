
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__trsmf) || defined(__cblas_trsm)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_solve_trm)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__trsmf)
void __trsmf(char *side, char *uplo, char *transa, char *diag,int *m, int *n,
             DTYPE *alpha, DTYPE *A, int *lda, DTYPE *B, int *ldb)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t a, b;
    int flags = 0;

    __armas_make(&b, *m, *n, *ldb, B);

    switch (toupper(*side)) {
    case 'R':
        flags |= ARMAS_RIGHT;
        __armas_make(&a, *n, *n, *lda, A);
        break;
    case 'L':
    default:
        flags |= ARMAS_LEFT;
        __armas_make(&a, *m, *m, *lda, A);
        break;
    }
    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;
    if (toupper(*transa) == 'T')
        flags |=  ARMAS_TRANS;
    if (toupper(*diag) == 'U')
        flags |=  ARMAS_UNIT;

    __armas_solve_trm(&b, &a, *alpha, flags, conf);
}
#endif

#if defined(__cblas_trsm)
void __cblas_trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                  const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa,
                  const enum CBLAS_DIAG diag, int M, int N,
                  DTYPE alpha, DTYPE *A, int lda, DTYPE *B, int ldb)
{
    armas_conf_t conf = *armas_conf_default();
    __armas_dense_t Aa, Ba;
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
        __armas_make(&Ba, M, N, ldb, B);
        if (side == CblasRight) {
            // N > lda --> error
            __armas_make(&Aa, N, N, lda, A);
        } else {
            // M > lda --> error
            __armas_make(&Aa, M, M, lda, A);
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
        __armas_make(&Ba, N, M, ldb, B);
        if (side == CblasRight) {
            // N > lda --> error
            __armas_make(&Aa, M, M, lda, A);
        } else {
            // M > lda --> error
            __armas_make(&Aa, N, N, lda, A);
        }
        break;
    default:
        return;
    }
    __armas_solve_trm(&Ba, &Aa, alpha, flags, &conf);
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
