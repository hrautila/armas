
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__nrm2f) || defined(__cblas_nrm2)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(COMPAT) && defined(__armas_nrm2)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__nrm2f)
DTYPE __nrm2f(int *n, DTYPE *X, int *incx)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t x;

    if (*incx == 1) {
        __armas_make(&x, *n, 1, *n, X);
    } else {
        __armas_make(&x, 1, *n, *incx, X);
    }
    return __armas_nrm2(&x, conf);
}
#endif

#if defined(__cblas_nrm2)
DTYPE __cblas_nrm2(int N, DTYPE *X, int incx)
{
    armas_conf_t *conf = armas_conf_default();
    __armas_dense_t x;

    if (incx == 1) {
        __armas_make(&x, N, 1, N, X);
    } else {
        __armas_make(&x, 1, N, incx, X);
    }
    return __armas_nrm2(&x, conf);
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
