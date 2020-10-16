
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_gemmf) || defined(cblas_gemm)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_mult)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_gemmf)
void blas_gemmf(char *transa, char *transb, int *m, int *n, int *k,
                DTYPE *alpha, DTYPE * A, int *lda, DTYPE * B, int *ldb,
                DTYPE * beta, DTYPE * C, int *ldc)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t c, a, b;
    int flags = 0;

    armas_make(&c, *m, *n, *ldc, C);
    armas_make(&a, *m, *k, *lda, A);
    armas_make(&b, *k, *n, *ldb, B);

    if (toupper(*transa) == 'T')
        flags |= ARMAS_TRANSA;
    if (toupper(*transb) == 'T')
        flags |= ARMAS_TRANSB;

    armas_mult(*beta, &c, *alpha, &a, &b, flags, conf);
}
#endif

#if defined(cblas_gemm)
void cblas_gemm(int order, int transa, int transb, int M, int N,
                int K, DTYPE alpha, DTYPE * A, int lda, DTYPE * B, int ldb,
                DTYPE beta, DTYPE * C, int ldc)
{
    armas_conf_t conf = *armas_conf_default();
    armas_dense_t Ca, Aa, Ba;
    int flags = 0;

    switch (order) {
    case CblasColMajor:
        if (transa == CblasTrans) {
            flags |= ARMAS_TRANSA;
            // error: K > lda
            armas_make(&Aa, K, M, lda, A);
        } else {
            // error: M > lda
            armas_make(&Aa, M, K, lda, A);
        }
        if (transb == CblasTrans) {
            flags |= ARMAS_TRANSB;
            armas_make(&Ba, N, K, ldb, B);
        } else {
            armas_make(&Ba, K, N, ldb, B);
        }
        armas_make(&Ca, M, N, ldc, C);
        break;
    case CblasRowMajor:
        if (transa == CblasNoTrans) {
            flags |= ARMAS_TRANSA;
            // error: M > lda
            armas_make(&Aa, M, K, lda, A);
        } else {
            // error: K > lda
            armas_make(&Aa, K, M, lda, A);
        }
        if (transb == CblasNoTrans) {
            flags |= ARMAS_TRANSB;
            armas_make(&Ba, K, N, ldb, B);
        } else {
            armas_make(&Ba, N, K, ldb, B);
        }
        armas_make(&Ca, N, M, ldc, C);
        break;
    default:
        return;
    }
    armas_mult(beta, &Ca, alpha, &Aa, &Ba, flags, &conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
