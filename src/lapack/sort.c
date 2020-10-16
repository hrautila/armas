
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_abs_sort_vec) && defined(armas_sort_vec) && defined(armas_sort_eigenvec)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_swap)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

// internal routines for sorting vectors and related matrices, like eigenvector matrices.
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

/**
 * @brief Sort vector on absolute values.
 *
 * @param[in,out] D
 *   On entry unorder vector. On exit elements sorted on absolute values
 *   as requested.
 * @param[in] updown
 *   Sort to ascending order if updown > 0, and to descending if < 0.
 *
 * Implementation is simple insertion sort.
 * @ingroup lapackaux
 */
void armas_abs_sort_vec(armas_dense_t * D, int updown)
{
    int k, j;
    DTYPE cval, tmpval;

    // simple insertion sort
    for (k = 1; k < armas_size(D); k++) {
        cval = ABS(armas_get_at_unsafe(D, k));
        for (j = k; j > 0; j--) {
            tmpval = ABS(armas_get_at_unsafe(D, j - 1));
            if (updown > 0 && tmpval >= cval) {
                break;
            }
            if (updown < 0 && tmpval <= cval) {
                break;
            }
            armas_set_at_unsafe(D, j, tmpval);
        }
        armas_set_at_unsafe(D, j, cval);
    }
}

/**
 * @brief Sort vector.
 *
 * @param[in,out] D
 *  Unorderd vector. On exit sorted vector.
 * @param[in] updown
 *  Sort to ascending order if updown > 0, and to descending if < 0.
 *
 * Implementation is simple insertion sort.
 * @ingroup lapackaux
 */
void armas_sort_vec(armas_dense_t * D, int updown)
{
    int k, j;
    DTYPE cval, tmpval;

    // simple insertion sort
    for (k = 1; k < armas_size(D); k++) {
        cval = armas_get_at_unsafe(D, k);
        for (j = k; j > 0; j--) {
            tmpval = armas_get_at_unsafe(D, j - 1);
            if (updown > 0 && tmpval >= cval) {
                break;
            }
            if (updown < 0 && tmpval <= cval) {
                break;
            }
            armas_set_at_unsafe(D, j, tmpval);
        }
        armas_set_at_unsafe(D, j, cval);
    }
}

/*
 * \brief Find minumum or maximum absolute value in vector
 */
static
int vec_minmax(armas_dense_t * D, int updown)
{
    int k, ix, incx, n;
    DTYPE cval, tmpval, *data;
    incx = D->rows == 1 ? D->step : 1;
    data = armas_data(D);
    cval = data[0];
    ix = 0;
    for (k = 1, n = incx; k < armas_size(D); k++, n += incx) {
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
static
int vec_abs_minmax(armas_dense_t * D, int minmax)
{
    int k, ix, n, incx;
    DTYPE cval, tmpval, *data;

    incx = D->rows == 1 ? D->step : 1;
    data = armas_data(D);
    cval = ABS(data[0]);
    ix = 0;
    for (k = 1, n = incx; k < armas_size(D); k++, n += incx) {
        tmpval = ABS(data[n]);
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

/**
 * @brief Sort eigenvalues and optionally related eigenvectors
 *
 * @param[in,out] D
 *      Eigenvalues, on exit sorted eigenvalues
 * @param[in,out] U
 *      Optional matrix of column eigenvectors, on exit columns sorted to reflect
 *      sorted eigenvalues.
 * @param[in,out] V
 *      Optional matrix of row eigenvectors, on exit rows sorted to reflect sorted eigenvalues.
 * @param[in,out] C
 *      Optional column matrix or vector, on exit columns/elements sorted to reflect sorted
 *      eigenvalues.
 * @param[in] updown
 *      Sort to ascending order if updown > 0 and descending order if < 0.
 *
 * @retval  0  Success
 * @retval <0 Failure. Returned if D is not vector.
 * @ingroup lapackaux
 */
int armas_sort_eigenvec(armas_dense_t * D, armas_dense_t * U,
                          armas_dense_t * V, armas_dense_t * C, int updown)
{
    DTYPE t0;
    int k, pk, N = armas_size(D);
    armas_dense_t sD, m0, m1;

    EMPTY(sD);

    if (!armas_isvector(D)) {
        return -ARMAS_ENEED_VECTOR;
    }
    // This is simple insertion sort - find index to largest/smallest value
    // in remaining subvector and swap that with value in current index.
    for (k = 0; k < N - 1; k++) {
        armas_subvector(&sD, D, k, N - k);
        pk = vec_minmax(&sD, -updown);
        if (pk != 0) {
            t0 = armas_get_at_unsafe(D, k);
            armas_set_at_unsafe(D, k, armas_get_at_unsafe(D, k + pk));
            armas_set_at_unsafe(D, pk + k, t0);
            if (U) {
                armas_column(&m0, U, k);
                armas_column(&m1, U, k + pk);
                armas_swap(&m1, &m0, (armas_conf_t *) 0);
            }
            if (V) {
                armas_row(&m0, V, k);
                armas_row(&m1, V, k + pk);
                armas_swap(&m1, &m0, (armas_conf_t *) 0);
            }
            if (C) {
                if (armas_isvector(C)) {
                    t0 = armas_get_at_unsafe(C, k);
                    armas_set_at_unsafe(C, k,
                                          armas_get_at_unsafe(C, k + pk));
                    armas_set_at_unsafe(C, pk + k, t0);

                } else {
                    armas_column(&m0, C, k);
                    armas_column(&m1, C, k + pk);
                    armas_swap(&m1, &m0, (armas_conf_t *) 0);
                }
            }
        }
    }
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
