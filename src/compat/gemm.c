
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__gemm) || defined(__cblas_gemm)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mult)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__gemm)
void __gemm(char *transa, char *transb, int *m, int *n, int *k, DTYPE *alpha, DTYPE *A,
            int *lda, DTYPE *B, int *ldb, DTYPE *beta, DTYPE *C, int *ldc)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t c, a, b;
    int flags = 0;

    __armas_make(&c, *m, *n, *ldc, C);
    __armas_make(&a, *m, *k, *lda, A);
    __armas_make(&b, *k, *n, *ldb, B);

    if (toupper(*transa) == 'T') 
        flags |= ARMAS_TRANSA;
    if (toupper(*transb) == 'T')
        flags |= ARMAS_TRANSB;

    __armas_mult(&c, &a, &b, *alpha, *beta, flags, conf);
}
#endif

#if defined(__cblas_gemm)
void __cblas_gemm(int order, int transa,  int transb, int M, int N,
                  int K, DTYPE alpha, DTYPE *A, int lda, DTYPE *B,  int ldb,
                  DTYPE beta, DTYPE *C, int ldc)
{
    armas_conf_t conf = *armas_conf_default();
    __armas_dense_t Ca, Aa, Ba;
    int flags = 0;
    
    switch (order) {
    case CblasColMajor:
        if (transa == CblasTrans) {
            flags |= ARMAS_TRANSA;
            // error: K > lda
            __armas_make(&Aa, K, M, lda, A);
        } else {
            // error: M > lda
            __armas_make(&Aa, M, K, lda, A);
        }
        if (transb == CblasTrans) {
            flags |= ARMAS_TRANSB;
            __armas_make(&Ba, N, K, ldb, B);
        } else {
            __armas_make(&Ba, K, N, ldb, B);
        }
        __armas_make(&Ca, M, N, ldc, C);
        break;
    case CblasRowMajor:
        if (transa == CblasNoTrans) {
            flags |= ARMAS_TRANSA;
            // error: M > lda
            __armas_make(&Aa, M, K, lda, A);
        } else {
            // error: K > lda
            __armas_make(&Aa, K, M, lda, A);
        }
        if (transb == CblasNoTrans) {
            flags |= ARMAS_TRANSB;
            __armas_make(&Ba, K, N, ldb, B);
        } else {
            __armas_make(&Ba, N, K, ldb, B);
        }
        __armas_make(&Ca, N, M, ldc, C);
        break;
    default:
        return;
    }
    __armas_mult(&Ca, &Aa, &Ba, alpha, beta, flags, &conf);
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
