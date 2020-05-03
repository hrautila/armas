
// Copyright (c) Harri Rautila, 2014-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(lapack_potrff) || defined(lapacke_potrf)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_cholfactor)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(lapack_potrff)
void lapack_potrff(char *uplo, int *n, DTYPE * A, int *lda, int *info)
{
    armas_x_dense_t a;
    armas_conf_t conf = *armas_conf_default();
    int err, flags = 0;

    armas_x_make(&a, *n, *n, *lda, A);
    flags = toupper(*uplo) == 'L' | ARMAS_LOWER : ARMAS_UPPER;
    err = armas_x_cholfactor(&a, flags, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(lapacke_potrf)
int lapacke_potrf(int order, char uplo, int n, DTYPE * A, int lda)
{
    armas_x_dense_t a;
    armas_conf_t conf = *armas_conf_default();
    int flags = 0;

    armas_x_make(&a, n, n, lda, A);
    if (order == LAPACKE_ROW_MAJOR) {
        flags = uplo == CblasUpper ? ARMAS_LOWER : ARMAS_UPPER;
    } else {
        flags = uplo == CblasLower ? ARMAS_LOWER : ARMAS_UPPER;
    }
    return armas_x_cholfactor(&a, flags, &conf);
}
#endif

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
