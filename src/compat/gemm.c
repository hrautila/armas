
// Copyright (c) Harri Rautila, 2012-2014

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

#if defined(COMPAT) && defined(__gemm)
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

#if defined(COMPAT_CBLAS) && defined(__cblas_gemm)
void __cblas_gemm(int order, int transa,  int transb, int M, int N,
                  int K, DTYPE alpha, DTYPE *A, int lda, DTYPE *B,  int ldb,
                  DTYPE beta, DTYPE *C, int ldc)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t c, a, b;
    int flags = 0;
    
    switch (order) {
    case CBLAS_ROW_MAJOR:
        flags = ARMAS_TRANSA|ARMAS_TRANSB;
        __armas_make(&c, *m, *n, *ldc, C);
        __armas_make(&a, *m, *k, *lda, A);
        __armas_make(&b, *k, *n, *ldb, B);
        if (transa == CBLAS_TRANS)
            flags |= ^ARMAS_TRANSB;
        if (transb == CBLAS_TRANS)
            flags |= ^ARMAS_TRANSA;
        // C.T = (B.T)*(A.T)
        __armas_mult(&c, &b, &a, *alpha, *beta, flags, conf);
        break;

    case CBLAS_COL_MAJOR:
    default:
        __armas_make(&c, *m, *n, *ldc, C);
        __armas_make(&a, *m, *k, *lda, A);
        __armas_make(&b, *k, *n, *ldb, B);
        if (transa == CBLAS_TRANS)
            flags |= ARMAS_TRANSA;
        if (transb == CBLAS_TRANS)
            flags |= ARMAS_TRANSB;
        
        __armas_mult(&c, &a, &b, *alpha, *beta, flags, conf);
        break;
    }
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
