
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#define __ARMAS_PROVIDES 1
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

// calculate maximum blocking size for an operation. 
int compute_lb(int M, int N, int wsz, WSSIZE worksize)
{
  int lb = 0;
  int k = 0;
  int wblk = (*worksize)(M, N, 4);
  if (wsz < wblk) {
    return 0;
  }
  do {
    lb += 4;
    wblk = (*worksize)(M, N, lb+4);
    k++;
  } while  (wsz > wblk && k < 100);
  if (k == 100)
    return 0;

  if (wblk > wsz) {
    lb -= 4;
  }
  return lb < 0 ? 0 : lb;
}

#if defined(__eigen_sort) && defined(__armas_iamax) && defined(__armas_swap)
/*
 * \brief Sort eigenvalues to decreasing order.
 *
 * Sorts eigenvalues (or singular values) in vector D to decreasing order and
 * rearrangens optional left and right eigenvector (singular vectors) to
 * corresponding order.
 *
 * \param[in,out] D
 *      Eigenvalue (singular value) vector
 * \param[in,out] U, V
 *      Left and right eigenvectors (singular vectors)
 * \param[in,out] C
 *      Optional matrix present in value U*C
 *
 */
int __eigen_sort(__armas_dense_t *D, __armas_dense_t *U,
                  __armas_dense_t *V, __armas_dense_t *C, armas_conf_t *conf)
{
    DTYPE t0;
    int k, pk, N = __armas_size(D);
    __armas_dense_t sD, m0, m1;

    if (! __armas_isvector(D)) {
        return -1;
    }

    // This is simple insertion sort - find index to largest value
    // in remaining subvector and swap that with value in current index.
    for (k = 0; k < N-1; k++) {
        __armas_subvector(&sD, D, k, N-k);
        pk = __armas_iamax(&sD, (armas_conf_t*)0);
        if (pk != 0) {
            t0 = __armas_get_at_unsafe(D, k);
            __armas_set_at_unsafe(D, k, __armas_get_at_unsafe(D, k+pk));
            __armas_set_at_unsafe(D, pk+k, t0);
            if (U) {
                __armas_column(&m0, U, k);
                __armas_column(&m1, U, k+pk);
                __armas_swap(&m1, &m0, conf);
            }
            if (C) {
                __armas_column(&m0, C, k);
                __armas_column(&m1, C, k+pk);
                __armas_swap(&m1, &m0, conf);
            }
            if (V) {
                __armas_row(&m0, V, k);
                __armas_row(&m1, V, k+pk);
                __armas_swap(&m1, &m0, conf);
            }
        }
    }
    return 0;
}
#endif // defined(__eigen_sort)

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// indent-tabs-mode: nil
// End:
