
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(blas_asumf) || defined(cblas_asum)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_asum)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_asumf)
DTYPE blas_asumf(int *n, DTYPE * X, int *incx)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t x;

    if (*incx == 1) {
        armas_make(&x, *n, 1, *n, X);
    } else {
        armas_make(&x, 1, *n, *incx, X);
    }
    return armas_asum(&x, conf);
}
#endif

#if defined(cblas_asum)
DTYPE cblas_asum(const int N, DTYPE * X, const int incx)
{
    armas_conf_t *conf = armas_conf_default();
    armas_dense_t x;

    if (incx == 1) {
        armas_make(&x, N, 1, N, X);
    } else {
        armas_make(&x, 1, N, incx, X);
    }
    return armas_asum(&x, conf);
}
#endif

#endif                          /* ARMAS_PROVIDES && ARMAS_REQUIRES */
