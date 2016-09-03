
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

#if defined(__eigen_sort) && defined(armas_x_iamax) && defined(armas_x_swap)
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
int __eigen_sort(armas_x_dense_t *D, armas_x_dense_t *U,
                  armas_x_dense_t *V, armas_x_dense_t *C, armas_conf_t *conf)
{
    DTYPE t0;
    int k, pk, N = armas_x_size(D);
    armas_x_dense_t sD, m0, m1;

    if (! armas_x_isvector(D)) {
        return -1;
    }

    // This is simple insertion sort - find index to largest value
    // in remaining subvector and swap that with value in current index.
    for (k = 0; k < N-1; k++) {
        armas_x_subvector(&sD, D, k, N-k);
        pk = armas_x_iamax(&sD, (armas_conf_t*)0);
        if (pk != 0) {
            t0 = armas_x_get_at_unsafe(D, k);
            armas_x_set_at_unsafe(D, k, armas_x_get_at_unsafe(D, k+pk));
            armas_x_set_at_unsafe(D, pk+k, t0);
            if (U) {
                armas_x_column(&m0, U, k);
                armas_x_column(&m1, U, k+pk);
                armas_x_swap(&m1, &m0, conf);
            }
            if (C) {
                armas_x_column(&m0, C, k);
                armas_x_column(&m1, C, k+pk);
                armas_x_swap(&m1, &m0, conf);
            }
            if (V) {
                armas_x_row(&m0, V, k);
                armas_x_row(&m1, V, k+pk);
                armas_x_swap(&m1, &m0, conf);
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
