
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(blas_getrff)  || defined(lapacke_getrf)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_lufactor)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(blas_getrff)
void blas_getrff(int *m, int *n, DTYPE * A, int *lda, int *ipiv, int *info)
{
    armas_x_dense_t a;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = imin(*m, *n);

    armas_x_make(&a, *m, *n, *lda, A);
    armas_pivot_make(&piv, npiv, ipiv);
    err = armas_x_lufactor(&a, &piv, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(lapacke_getrf)
int lapacke_getrf(int order, int M, int N, DTYPE * A, int lda, int *ipv)
{
    armas_x_dense_t Aa;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = imin(M, N);

    if (order == LAPACKE_ROW_MAJOR) {
        return -1;
    }
    armas_x_make(&Aa, M, N, lda, A);
    armas_pivot_make(&piv, npiv, ipv);
    err = armas_x_lufactor(&Aa, &piv, &conf);
    return err ? -conf.error : 0;
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
