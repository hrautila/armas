
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "compat.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(__potrsf) || defined(__cblas_potrs) || defined(__lapacke_potrs)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_cholsolve)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------
#include <ctype.h>
#include "matrix.h"

#if defined(__potrsf)
void __potrsf(char *uplo, int *n, int *nrhs, DTYPE * A, int *lda, DTYPE * B,
              int *ldb, int *info)
{
    armas_dense_t a, b;
    armas_conf_t conf = *armas_conf_default();
    int err, flags = 0;

    armas_make(&a, *n, *n, *lda, A);
    armas_make(&b, *n, *nrhs, *ldb, B);
  flags = toupper(*uplo) == 'L' | ARMAS_LOWER:ARMAS_UPPER;
    err = armas_cholsolve(&b, &a, flags, &conf);
    *info = err ? -conf.error : 0;
}
#endif

#if defined(__cblas_potrs)
int __cblas_potrs(int order, int uplo, int n, DTYPE * A, int lda, DTYPE * B,
                  int ldb)
{
}
#endif

#if defined(__lapacke_potrs)
int __lapacke_potrs(int order, int uplo, int n, int nrhs, DTYPE * A, int lda,
                    DTYPE * B, int ldb)
{
    armas_dense_t a, b;
    armas_conf_t conf = *armas_conf_default();
    int err, flags = 0;

    if (order == LAPACK_ROW_MAJOR) {
        // needs copying; not yet implemented
        return -1;
    } else {
        armas_make(&a, n, n, lda, A);
        armas_make(&b, n, nrhs, ldb, B);
      flags = toupper(uplo) == 'L' | ARMAS_LOWER:ARMAS_UPPER;
    }
    err = armas_cholsolve(&b, &a, flags, &conf);
    return err ? -conf.error : 0;
}
#endif

#endif                          /* ARMAS_PROVIDES && ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
