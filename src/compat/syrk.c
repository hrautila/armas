
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_syrkf) || defined(cblas_syrk)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_update_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_syrkf)
void blas_syrkf(char *uplo, char *trans, int *n, int *k, DTYPE * alpha,
                DTYPE * A, int *lda, DTYPE * beta, DTYPE * C, int *ldc)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t c, a;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;
    if (toupper(*trans) == 'T')
        flags |= ARMAS_TRANS;

    armas_make(&c, *n, *n, *ldc, C);
    if (flags & ARMAS_TRANS) {
        armas_make(&a, *k, *n, *lda, A);
    } else {
        armas_make(&a, *n, *k, *lda, A);
    }
    armas_update_sym(*beta, &c, *alpha, &a, flags, conf);
}
#endif

#if defined(cblas_syrk)
void cblas_syrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                const enum CBLAS_TRANSPOSE trans, int N, int K,
                DTYPE alpha, DTYPE * A, int lda, DTYPE beta, DTYPE * C,
                int ldc)
{
    armas_conf_t conf = *armas_conf_default();
    armas_dense_t Ca, Aa;
    int flags = 0;

    switch (order) {
    case CblasRowMajor:
        flags |= uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
        if (trans == CblasNoTrans) {
            flags |= ARMAS_TRANS;
            armas_make(&Aa, K, N, lda, A);
        } else {
            armas_make(&Aa, N, K, lda, A);
        }
        break;
    case CblasColMajor:
    default:
        flags |= uplo == CblasUpper ? ARMAS_UPPER : ARMAS_LOWER;
        if (trans == CblasTrans) {
            flags |= ARMAS_TRANS;
            armas_make(&Aa, K, N, lda, A);
        } else {
            armas_make(&Aa, N, K, lda, A);
        }
        break;
    }
    armas_make(&Ca, N, N, ldc, C);
    armas_update_sym(beta, &Ca, alpha, &Aa, flags, conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
