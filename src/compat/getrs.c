
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__getrs)  || defined(__lapacke_getrs)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(COMPAT) && defined(__armas_lufactor)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__getrs)
void __getrs(int *n, int *nrhs, DTYPE *A, int *lda, int *ipiv, DTYPE *B, int *ldb, int *info)
{
    __armas_dense_t a, b;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = *n;

    __armas_make(&a, *n, *n, *lda, A);
    __armas_make(&b, *n, *nrhs, *ldb, B);
    armas_pivot_make(&piv, npiv, ipiv);
    err = __armas_lusolve(&b, &a, &piv, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(__lapacke_getrs)
int __lapacke_getrs(int order, int N, int NRHS, DTYPE *A, int lda, int *ipv, DTYPE *B, int ldb)
{
    __armas_dense_t Aa, Ba;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = imin(M, N);

    if (order == LAPACKE_ROW_MAJOR) {
        return -1;
    }
    __armas_make(&Aa, M, N, lda, A);
    __armas_make(&Ba, N, NRHS, ldb, B);
    armas_pivot_make(&piv, npiv, ipv);
    err = __armas_lusolve(&Ba, &Aa, &piv, &conf);
    return err ? -conf.error : 0;
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
