
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__ger) || defined(__cblas_ger)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_mvupdate)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) && defined(__ger)
void __ger(int *m, int *n, DTYPE *alpha, DTYPE *X,
            int *incx, DTYPE *Y, int *incy, DTYPE *A, int *lda)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t y, a, x;

    __armas_make(&a, *m, *n, *lda, A);
    if (*incy == 1) {
        // column vector
        __armas_make(&y, *n, 1, *n, Y);
    } else {
        // row vector
        __armas_make(&y, 1, *n, *incy, Y);
    }
    if (*incx == 1) {
        __armas_make(&x, *m, 1, *m, X);
    } else {
        __armas_make(&x, 1, *m, *incx, X);
    }
    __armas_mvupdate(&a, &x, &y, *alpha, conf);
}
#endif

#if defined(COMPAT_CBLAS) && defined(__cblas_ger)
void __cblas_ger(int order, int uplo,  int N, DTYPE alpha,
                  DTYPE *X, int incx, DTYPE *Y, int incy, DTYPE *A, int lda)
{
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
