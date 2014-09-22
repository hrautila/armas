
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__syr2k) || defined(__cblas_syr2k)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_update2_sym)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) && defined(__syr2k)
void __syr2k(char *uplo, char *trans, int *n, int *k, DTYPE *alpha, DTYPE *A,
             int *lda, DTYPE *B, int *ldb, DTYPE *beta, DTYPE *C, int *ldc)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t c, a, b;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;
    if (toupper(*trans) == 'T') 
        flags |= ARMAS_TRANS;

    __armas_make(&c, *n, *n, *ldc, C);
    if (flags & ARMAS_TRANS) {
        __armas_make(&a, *k, *n, *lda, A);
        __armas_make(&b, *k, *n, *lda, B);
    } else {
        __armas_make(&a, *n, *k, *lda, A);
        __armas_make(&b, *n, *k, *lda, B);
    }

    __armas_update2_sym(&c, &a, &b, *alpha, *beta, flags, conf);
}
#endif

#if defined(COMPAT_CBLAS) && defined(__cblas_syr2k)
void __cblas_syr2k(int order, int uplo,  int trans, int N, int K, DTYPE alpha,
                   DTYPE *A, int lda, DTYPE *B, int ldb, DTYPE beta, DTYPE *C, int ldc)
{
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
