
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(blas_syr2kf) || defined(cblas_syr2k)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_update2_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_syr2kf)
void blas_syr2kf(char *uplo, char *trans, int *n, int *k, DTYPE * alpha,
                 DTYPE * A, int *lda, DTYPE * B, int *ldb, DTYPE * beta,
                 DTYPE * C, int *ldc)
{
    armas_conf_t *conf = armas_conf_default();
    armas_x_dense_t c, a, b;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;
    if (toupper(*trans) == 'T')
        flags |= ARMAS_TRANS;

    armas_x_make(&c, *n, *n, *ldc, C);
    if (flags & ARMAS_TRANS) {
        armas_x_make(&a, *k, *n, *lda, A);
        armas_x_make(&b, *k, *n, *lda, B);
    } else {
        armas_x_make(&a, *n, *k, *lda, A);
        armas_x_make(&b, *n, *k, *lda, B);
    }
    armas_x_update2_sym(*beta, &c, *alpha, &a, &b, flags, conf);
}
#endif

#if defined(cblas_syr2k)
void cblas_syr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, int N, int K, DTYPE alpha,
                 DTYPE * A, int lda, DTYPE * B, int ldb, DTYPE beta,
                 DTYPE * C, int ldc)
{
    armas_conf_t conf = *armas_conf_default();
    armas_x_dense_t Ca, Aa, Ba;
    int flags = 0;

    switch (order) {
    case CblasRowMajor:
        flags |= uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        if (trans == CblasNoTrans) {
            flags |= ARMAS_TRANS;
            armas_x_make(&Aa, K, N, lda, A);
        } else {
            armas_x_make(&Aa, N, K, lda, A);
        }
        break;
    case CblasColMajor:
    default:
        flags |= uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        if (trans == CblasTrans) {
            flags |= ARMAS_TRANS;
            armas_x_make(&Aa, K, N, lda, A);
            armas_x_make(&Ba, K, N, ldb, B);
        } else {
            armas_x_make(&Aa, N, K, lda, A);
            armas_x_make(&Ba, N, K, ldb, B);
        }
        break;
    }
    armas_x_make(&Ca, N, N, ldc, C);
    armas_x_update2_sym(beta, &Ca, alpha, &Aa, &Ba, flags, conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
