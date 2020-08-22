
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(lapack_getrsf)  || defined(lapacke_getrs)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_lusolve)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(lapack_getrsf)
void lapack_getrsf(int *n, int *nrhs, DTYPE * A, int *lda, int *ipiv, DTYPE * B,
                   int *ldb, int *info)
{
    armas_x_dense_t a, b;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = *n;

    armas_x_make(&a, *n, *n, *lda, A);
    armas_x_make(&b, *n, *nrhs, *ldb, B);
    armas_pivot_make(&piv, npiv, ipiv);
    err = armas_x_lusolve(&b, &a, &piv, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(lapacke_getrs)
int lapacke_getrs(int order, int N, int NRHS, DTYPE * A, int lda, int *ipv,
                  DTYPE * B, int ldb)
{
    armas_x_dense_t Aa, Ba;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = imin(M, N);

    if (order == LAPACKE_ROW_MAJOR) {
        return -1;
    }
    armas_x_make(&Aa, M, N, lda, A);
    armas_x_make(&Ba, N, NRHS, ldb, B);
    armas_pivot_make(&piv, npiv, ipv);
    err = armas_x_lusolve(&Ba, &Aa, &piv, &conf);
    return err ? -conf.error : 0;
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
