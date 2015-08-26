
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__trmm) || defined(__cblas_trmm)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mult_trm)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) && defined(__trmm)
void __trmm(char *side, char *uplo, char *transa, char *diag,int *m, int *n,
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

    __armas_mult_trm(&b, &a, *alpha, flags, conf);
}
#endif

#if defined(COMPAT_CBLAS) && defined(__cblas_trmm)
void __cblas_trmm(int order, int side,  int uplo, int transa, int diag, int M, int N,
                  DTYPE alpha, DTYPE *A, int lda, DTYPE *B,  int ldb)
{
    printf("libarmas-compat.cblas_trmm: not implemented\n");
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
