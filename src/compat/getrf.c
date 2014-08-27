
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__getrf) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_lufactor)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(COMPAT) && defined(__getrf)
void __getrf(int *m, int *n, DTYPE *A, int *lda, int *ipiv, int *info)
{
    __armas_dense_t a;
    armas_pivot_t piv;
    armas_conf_t conf = *armas_conf_default();
    int err, npiv = imin(*m, *n);

    __armas_make(&a, *m, *n, *lda, A);
    armas_pivot_make(&piv, npiv, ipiv);
    err = __armas_lufactor(&a, &piv, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
