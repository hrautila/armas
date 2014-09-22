
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__abs_sort_vec) && defined(__sort_vec) && defined(__sort_eigenvec)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_swap) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

// internal routines for sorting vectors and related matrices, like eigenvector matrices.


#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"


/*
 * \brief Sort vector on absolute values.
 *
 * \param[in,out] D
 *      Vector to sort
 * \param[in] updown
 *      Sort to ascending order if updown > 0, and to descending if < 0.
 */
void __abs_sort_vec(__armas_dense_t *D, int updown) 
{
    int k,  j;
    DTYPE cval, tmpval;

    // simple insertion sort
    for (k = 1; k < __armas_size(D); k++) {
        cval = __ABS(__armas_get_at_unsafe(D, k));
        for (j = k; j > 0; j--) {
            tmpval = __ABS(__armas_get_at_unsafe(D, j-1));
            if (updown > 0 && tmpval >= cval) {
                break;
            }
            if (updown < 0 && tmpval <= cval) {
                break;
            }
            __armas_set_at_unsafe(D, j, tmpval);
        }
        __armas_set_at_unsafe(D, j, cval);
    }
}

/*
 * \brief Sort vector.
 *
 * \param[in,out] D
 *      Vector to sort
 * \param[in] updown
 *      Sort to ascending order if updown > 0, and to descending if < 0.
 */
void __sort_vec(__armas_dense_t *D, int updown) 
{
    int k, j;
    DTYPE cval, tmpval;

    // simple insertion sort
    for (k = 1; k < __armas_size(D); k++) {
        cval = __armas_get_at_unsafe(D, k);
        for (j = k; j > 0; j--) {
            tmpval = __armas_get_at_unsafe(D, j-1);
            if (updown > 0 && tmpval >= cval) {
                break;
            }
            if (updown < 0 && tmpval <= cval) {
                break;
            }
            __armas_set_at_unsafe(D, j, tmpval);
        }
        __armas_set_at_unsafe(D, j, cval);
    }
}

/*
 * \brief Find minumum or maximum absolute value in vector
 */
static inline
int __vec_minmax(__armas_dense_t *D, int updown)
{
    int k, ix, incx, n;
    DTYPE cval, tmpval, *data;
    incx = D->rows == 1 ? D->step : 1;
    data = __armas_data(D);
    cval = data[0];
    ix = 0;
    for (k = 1, n = incx; k < __armas_size(D); k++, n += incx) {
        tmpval = data[n];
        if (updown > 0 && tmpval > cval) {
            cval = tmpval;
            ix = k;
        }
        if (updown < 0 && tmpval < cval) {
            cval = tmpval;
            ix = k;
        }
    }
    return ix;
}

/*
 * \brief Find minumum or maximum absolute value in vector
 */
static inline
int __vec_abs_minmax(__armas_dense_t *D, int minmax)
{
    int k, ix, n, incx;
    DTYPE cval, tmpval, *data;

    incx = D->rows == 1 ? D->step : 1;
    data = __armas_data(D);
    cval = __ABS(data[0]);
    ix = 0;
    for (k = 1, n = incx ; k < __armas_size(D); k++, n += incx) {
        tmpval = __ABS(data[n]);
        if (minmax > 0 && tmpval > cval) {
            cval = tmpval;
            ix = k;
        }
        if (minmax < 0 && tmpval < cval) {
            cval = tmpval;
            ix = k;
        }
    }
    return ix;
}

/*
 * \brief Sort eigenvalues and optionally related eigenvectors
 *
 * \param[in,out] D
 *      Eigenvalues, on exit sorted eigenvalues
 * \param[in,out] U
 *      Optional matrix of column eigenvectors, on exit columns sorted to reflect
 *      sorted eigenvalues.
 * \param[in,out] V
 *      Optional matrix of row eigenvectors, on exit rows sorted to reflect sorted eigenvalues.
 * \param[in,out] C
 *      Optional column matrix, on exit rows sorted to reflect sorted eigenvalues.
 * \param[in] updown
 *      Sort to ascending order if updown > 0 and descending order if < 0.
 */
int __sort_eigenvec(__armas_dense_t *D, __armas_dense_t *U,
                    __armas_dense_t *V, __armas_dense_t *C, int updown)
{
    DTYPE t0;
    int k, pk, N = __armas_size(D);
    __armas_dense_t sD, m0, m1;

    if (! __armas_isvector(D)) {
        return -1;
    }

    // This is simple insertion sort - find index to largest/smallest value
    // in remaining subvector and swap that with value in current index.
    for (k = 0; k < N-1; k++) {
        __armas_subvector(&sD, D, k, N-k);
        pk = __vec_minmax(&sD, -updown);
        if (pk != 0) {
            t0 = __armas_get_at_unsafe(D, k);
            __armas_set_at_unsafe(D, k, __armas_get_at_unsafe(D, k+pk));
            __armas_set_at_unsafe(D, pk+k, t0);
            if (U) {
                __armas_column(&m0, U, k);
                __armas_column(&m1, U, k+pk);
                __armas_swap(&m1, &m0, (armas_conf_t *)0);
            }
            if (C) {
                __armas_column(&m0, C, k);
                __armas_column(&m1, C, k+pk);
                __armas_swap(&m1, &m0, (armas_conf_t *)0);
            }
            if (V) {
                __armas_row(&m0, V, k);
                __armas_row(&m1, V, k+pk);
                __armas_swap(&m1, &m0, (armas_conf_t *)0);
            }
        }
    }
    return 0;
}

#if 0
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
int __svd_sort(__armas_dense_t *D, __armas_dense_t *U,
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
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

