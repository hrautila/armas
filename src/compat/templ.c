
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__xxxx) || defined(__lapacke_xxxx)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_zzzz)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__xxxx)
void __xxxx(char *transa, char *transb, int *m, int *n, int *k, DTYPE *alpha, DTYPE *A,
            int *lda, DTYPE *B, int *ldb, DTYPE *beta, DTYPE *C, int *ldc)
{
}
#endif

#if defined(__lapacke_xxxx)
void __lapacke_xxxx(int order, int transa,  int transb, int M, int N,
                  int K, DTYPE alpha, DTYPE *A, int lda, DTYPE *B,  int ldb,
                  DTYPE beta, DTYPE *C, int ldc)
{
}

#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
