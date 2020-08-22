
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(blas_symmf) || defined(cblas_symm)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_symmf)
void blas_symmf(char *side, char *uplo, int *m, int *n, DTYPE * alpha,
                DTYPE * A, int *lda, DTYPE * B, int *ldb, DTYPE * beta,
                DTYPE * C, int *ldc)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t c, a, b;
    int flags = 0;

    armas_x_make(&c, *m, *n, *ldc, C);
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

    armas_x_mult_sym(*beta, &c, *alpha, &a, &b, flags, conf);
}
#endif

#if defined(cblas_symm)
void cblas_symm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                const enum CBLAS_UPLO uplo, int M, int N,
                DTYPE alpha, DTYPE * A, int lda, DTYPE * B, int ldb,
                DTYPE beta, DTYPE * C, int ldc)
{
    armas_conf_t conf = *armas_conf_default();
    armas_x_dense_t Ca, Aa, Ba;
    int flags = 0;

    switch (order) {
    case CblasRowMajor:
        if (side == CblasRight) {
            flags |= ARMAS_LEFT;
            armas_x_make(&Aa, N, N, lda, A);
        } else {
            flags |= ARMAS_RIGHT;
            armas_x_make(&Aa, M, M, lda, A);
        }
        armas_x_make(&Ba, N, M, ldb, B);
        armas_x_make(&Ca, N, M, ldc, C);
        flags |= uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        break;
    case CblasColMajor:
    default:
        if (side == CblasRight) {
            flags |= ARMAS_RIGHT;
            armas_x_make(&Aa, M, M, lda, A);
        } else {
            flags |= ARMAS_LEFT;
            armas_x_make(&Aa, N, N, lda, A);
        }
        armas_x_make(&Ba, M, N, ldb, B);
        armas_x_make(&Ca, M, N, ldc, C);
        flags |= uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        break;
    }
    armas_x_mult_sym(beta, &Ca, alpha, &Aa, &Ba, flags, &conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
