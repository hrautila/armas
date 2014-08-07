
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_bdbuild) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_qrbuild) && defined(__armas_lqbuild)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

/*
 * Generate one of the orthogonal matrices Q or P.T determined by BDReduce() when
 * reducing a real matrix A to bidiagonal form. Q and P.T are defined as products
 * elementary reflectors H(i) or G(i) respectively.
 *
 * Orthogonal matrix Q is generated if flag WANTQ is set. And matrix P respectively
 * of flag WANTP is set.
 */
int __armas_bdbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                    int K, int flags, armas_conf_t *conf)
{
    __armas_dense_t Qh, Ph, tauh, d, s;
    int j, err;

    if (!conf)
        conf = armas_conf_default();

    if (__armas_size(A) == 0)
        return 0;

    if (A->rows >= A->cols) {
        switch (flags & (ARMAS_WANTQ|ARMAS_WANTP)) {
        case ARMAS_WANTQ:
            __armas_submatrix(&tauh, tau, 0, 0, A->cols, 1);
            err = __armas_qrbuild(A, &tauh, W, K, conf);
            break;
        case ARMAS_WANTP:
            // shift P matrix embedded in A down and fill first column and
            // row to unit vectors
            for (j = A->cols-1; j > 0; j--) {
                __armas_submatrix(&s, A, j-1, j, 1, A->cols-j);
                __armas_submatrix(&d, A, j,   j, 1, A->cols-j);
                __armas_copy(&d, &s, conf);
                __armas_set(A, j, 0, __ZERO);
            }
            // zero first row
            __armas_row(&d, A, 0);
            __armas_scale(&d, __ZERO, conf);
            __armas_set(&d, 0, 0, __ONE);

            __armas_submatrix(&Ph, A, 1, 1, A->cols-1, A->cols-1);
            __armas_submatrix(&tauh, tau, 0, 0, A->cols-1, 1);
            if (K > A->cols-1 || K < 0)
                K = A->cols - 1;
            err = __armas_lqbuild(&Ph, &tauh, W, K, conf);
            break;
        default:
            break;
        }
    } else {
        switch (flags & (ARMAS_WANTQ|ARMAS_WANTP)) {
        case ARMAS_WANTQ:
            // shift Q matrix embedded in A right and fill first column and
            // row to unit vectors
            for (j = A->rows-1; j > 0; j--) {
                __armas_submatrix(&s, A, j, j-1, A->rows-j, 1);
                __armas_submatrix(&d, A, j, j,   A->rows-j, 1);
                __armas_copy(&d, &s, conf);
                __armas_set(A, 0, j, __ZERO);
            }
            // zero first column
            __armas_column(&d, A, 0);
            __armas_scale(&d, __ZERO, conf);
            __armas_set(&d, 0, 0, __ONE);

            __armas_submatrix(&Qh, A, 1, 1, A->rows-1, A->rows-1);
            __armas_submatrix(&tauh, tau, 0, 0, A->rows-1, 1);
            if (K > A->rows-1 || K < 0)
                K = A->rows - 1;
            err = __armas_qrbuild(&Qh, &tauh, W, K, conf);
            break;
        case ARMAS_WANTP:
            __armas_submatrix(&tauh, tau, 0, 0, A->rows, 1);
            err = __armas_lqbuild(A, &tauh, W, K, conf);
            break;
        default:
            break;
        }
    }
    return err;
}

int __armas_bdbuild_work(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
  if (!conf)
    conf = armas_conf_default();
  if (flags & ARMAS_WANTP) {
      return __armas_lqbuild_work(A, conf);
  }
  return __armas_qrbuild_work(A, conf);
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

