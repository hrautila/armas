
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_qsort_vec)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

// internal routines for sorting vectors and related matrices, like eigenvector matrices.
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#define CUTOFF 8

static inline
void armas_isort_asc(armas_dense_t *x)
{
    int k, j;
    DTYPE cval, tmpval;
    for (k = 1; k < armas_size(x); k++) {
        cval = armas_get_at_unsafe(x, k);
        for (j = k; j > 0; j--) {
            tmpval = armas_get_at_unsafe(x, j - 1);
            if (tmpval <= cval)
                break;
            armas_set_at_unsafe(x, j, tmpval);
        }
        armas_set_at_unsafe(x, j, cval);
    }
}

static inline
void armas_isort_desc(armas_dense_t *x)
{
    int k, j;
    DTYPE cval, tmpval;
    for (k = 1; k < armas_size(x); k++) {
        cval = armas_get_at_unsafe(x, k);
        for (j = k; j > 0; j--) {
            tmpval = armas_get_at_unsafe(x, j - 1);
            if (tmpval >= cval)
                break;
            armas_set_at_unsafe(x, j, tmpval);
        }
        armas_set_at_unsafe(x, j, cval);
    }
}

static
void armas_qsort_asc(armas_dense_t *x)
{
    int i, j, n;
    DTYPE pivot, tmp;
    armas_dense_t x0, x1;

    if (armas_size(x) < CUTOFF) {
        armas_isort_asc(x);
        return;
    }
    n = armas_size(x);
    pivot = armas_get_at_unsafe(x, n/2);
    armas_set_at_unsafe(x, n/2, armas_get_at_unsafe(x, 0));
    armas_set_at_unsafe(x, 0, pivot);
    i = 1; j = n - 1;
    for (;;) {
        for (; armas_get_at_unsafe(x, i) < pivot && i < n; i++);
        for (; armas_get_at_unsafe(x, j) > pivot; j--);
        if (i >= j)
            break;
        tmp = armas_get_at_unsafe(x, i);
        armas_set_at_unsafe(x, i, armas_get_at_unsafe(x, j));
        armas_set_at_unsafe(x, j, tmp);
    }
    armas_set_at_unsafe(x, 0, armas_get_at_unsafe(x, j));
    armas_set_at_unsafe(x, j, pivot);

    armas_subvector_unsafe(&x0, x, 0, j);
    armas_subvector_unsafe(&x1, x, j + 1, n - j - 1);

    armas_qsort_asc(&x0);
    armas_qsort_asc(&x1);
}

static
void armas_qsort_desc(armas_dense_t *x)
{
    int i, j, n;
    DTYPE pivot, tmp;
    armas_dense_t x0, x1;

    if (armas_size(x) < CUTOFF) {
        armas_isort_desc(x);
        return;
    }
    n = armas_size(x);
    pivot = armas_get_at_unsafe(x, n/2);
    armas_set_at_unsafe(x, n/2, armas_get_at_unsafe(x, 0));
    armas_set_at_unsafe(x, 0, pivot);
    i = 1; j = n - 1;
    for (;;) {
        for (; armas_get_at_unsafe(x, i) > pivot && i < n; i++);
        for (; armas_get_at_unsafe(x, j) < pivot; j--);
        if (i >= j)
            break;
        tmp = armas_get_at_unsafe(x, i);
        armas_set_at_unsafe(x, i, armas_get_at_unsafe(x, j));
        armas_set_at_unsafe(x, j, tmp);
    }
    armas_set_at_unsafe(x, 0, armas_get_at_unsafe(x, j));
    armas_set_at_unsafe(x, j, pivot);

    armas_subvector_unsafe(&x0, x, 0, j);
    armas_subvector_unsafe(&x1, x, j + 1, n - j - 1);

    armas_qsort_desc(&x0);
    armas_qsort_desc(&x1);
}


/**
 * @brief Sort vector with quicksort algorithm.
 *
 * @param[in,out] D
 *  Unorderd vector. On exit sorted vector.
 * @param[in] updown
 *  Sort to ascending order if updown > 0, and to descending if < 0.
 *
 * Implementation is quicksort.
 * @ingroup lapackaux
 */
int armas_qsort_vec(armas_dense_t * D, int updown)
{
    if (!armas_isvector(D)) {
        return -ARMAS_ENEED_VECTOR;
    }

    if (updown > 0) {
        armas_qsort_asc(D);
    } else {
        armas_qsort_desc(D);
    }
    return 0;
}

#else
#warning "Missing defines. No code!"
#endif
