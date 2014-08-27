
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__potrf) || defined(__cblas_potrf)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_cholfactor)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) && defined(__potrf)
void __potrf(char *uplo, int *n, DTYPE *A, int *lda, int *info)
{
    __armas_dense_t a;
    armas_conf_t conf = *armas_conf_default();
    int err, flags = 0;

    __armas_make(&a, *n, *n, *lda, A);
    flags = toupper(*uplo) == 'L' | ARMAS_LOWER : ARMAS_UPPER;
    err = __armas_cholfactor(&a, flags, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(COMPAT_CBLAS) && defined(__cblas_potrf)
int __cblas_potrf(int order, int uplo, int n, DTYPE *A, int lda)
{
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
