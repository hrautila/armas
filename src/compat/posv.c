
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__posv) || defined(__cblas_posv)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_cholfactor) && defined(__armas_cholsolve)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__posv)
void __posv(char *uplo, int *n, int *nrhs, DTYPE *A, int *lda, DTYPE *B, int *ldb,int *info)
{
    __armas_dense_t a, b;
    armas_conf_t conf = *armas_conf_default();
    int err, flags = 0;

    __armas_make(&a, *n, *n, *lda, A);
    __armas_make(&b, *n, *nrhs, *ldb, B);
    flags = toupper(*uplo) == 'L' ? ARMAS_LOWER : ARMAS_UPPER;
    err = __armas_cholfactor(&a, flags, &conf);
    if (!err)
        err = __armas_cholsolve(&b, &a, flags, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(__cblas_posv)
int __cblas_posv(int order, int uplo, int n, DTYPE *A, int lda, DTYPE *B, int ldb)
{
    __armas_dense_t Aa, Ba;
    armas_conf_t conf = *armas_conf_default();
    int err, flags = 0;

    if (order == LAPACKE_ROW_MAJOR) {
        // solving needs copying; not yet implemented
        return -1;
    }
    __armas_make(&Aa, n, n, lda, A);
    __armas_make(&Ba, n, nrhs, ldb, B);
    flags = uplo == CblasLower ? ARMAS_LOWER : ARMAS_UPPER;
    err = __armas_cholfactor(&Aa, flags, &conf);
    if (!err)
        err = __armas_cholsolve(&Ba, &Aa, flags, &conf);
    return err ? -conf.error : 0;
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
