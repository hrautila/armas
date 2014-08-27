
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__syr) || defined(__cblas_syr)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mvupdate_sym)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) && defined(__syr)
void __syr(char *uplo, int *n, DTYPE *alpha, DTYPE *X,
            int *incx, DTYPE *A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t a, x;
    int flags = 0;

    flags |= toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;

    __armas_make(&a, *n, *n, *lda, A);
    if (*incx == 1) {
        __armas_make(&x, *n, 1, *n, X);
    } else {
        __armas_make(&x, 1, *n, *incx, X);
    }
    __armas_mvupdate_sym(&a, &x, *alpha, flags, conf);
}
#endif

#if defined(COMPAT_CBLAS) && defined(__cblas_syr)
void __cblas_syr(int order, int uplo,  int N,
                 DTYPE alpha, DTYPE *X, int incx, DTYPE *A, int lda)
{
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
