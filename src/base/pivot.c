
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "armas.h"
#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_x_lapack_pivots)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#if defined(armas_x_swap) && defined(armas_x_iamax)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "pivot.h"

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 *
 * LOWER triangular; srcix < dstix
 *
 *    d
 *    x  d 
 *    S1 S1 P1 x  x  x  P2     -- src row/col 'srcix'
 *    x  x  S2 d  x  x  x
 *    x  x  S2 x  d  x  x
 *    x  x  S2 x  x  d  x
 *    D1 D1 P2 D2 D2 D2 P3     -- dest row/col 'dstix'
 *    x  x  S3 x  x  x  D3 d
 *    x  x  S3 x  x  x  D3 x d
 *         (AR)
 */
static
void apply_pivot_lower(armas_x_dense_t * A, int srcix, int dstix,
                       armas_conf_t * conf)
{
    armas_x_dense_t s, d;
    DTYPE p1, p3;

    if (srcix == dstix)
        return;
    if (srcix > dstix) {
        int t = srcix;
        srcix = dstix;
        dstix = t;
    }
    // S1 -> D1
    if (srcix > 0) {            // len(S1) > 0
        armas_x_submatrix_unsafe(&s, A, srcix, 0, 1, srcix);
        armas_x_submatrix_unsafe(&d, A, dstix, 0, 1, srcix);
        armas_x_swap(&s, &d, conf);
    }
    // S2 -> D2
    armas_x_submatrix_unsafe(&s, A, srcix + 1, srcix, dstix - srcix - 1, 1);
    armas_x_submatrix_unsafe(&d, A, dstix, srcix + 1, 1, dstix - srcix - 1);
    armas_x_swap(&s, &d, conf);
    // S3 -> D3
    armas_x_submatrix_unsafe(&s, A, dstix + 1, srcix, A->rows - dstix - 1, 1);
    armas_x_submatrix_unsafe(&d, A, dstix + 1, dstix, A->rows - dstix - 1, 1);
    armas_x_swap(&s, &d, conf);
    // swap P1 and P3
    p1 = armas_x_get_unsafe(A, srcix, srcix);
    p3 = armas_x_get_unsafe(A, dstix, dstix);
    armas_x_set_unsafe(A, srcix, srcix, p3);
    armas_x_set_unsafe(A, dstix, dstix, p1);
}

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 *
 * UPPER triangular; moving from bottom-right to top-left
 *
 *    srcix > dstix:
 *    d x D3 x  x  x  S3 x
 *      d D3 x  x  x  S3 x
 *        P3 D2 D2 D2 P2 D1   -- dstinx
 *           d  x  x  S2 x
 *              d  x  S2 x
 *                 d  S2 x
 *                    P1 S1   -- srcinx
 *                       d
 */
static
void apply_pivot_upper(armas_x_dense_t * A, int srcix, int dstix,
                       armas_conf_t * conf)
{
    armas_x_dense_t s, d;
    DTYPE p1, p3;

    if (srcix == dstix)
        return;
    if (srcix < dstix) {
        int t = srcix;
        srcix = dstix;
        dstix = t;
    }
    // S1 -- D1
    armas_x_submatrix_unsafe(&s, A, srcix, srcix + 1, 1, A->cols - srcix - 1);
    armas_x_submatrix_unsafe(&d, A, dstix, srcix + 1, 1, A->cols - srcix - 1);
    armas_x_swap(&s, &d, conf);
    // S2 -- D2
    armas_x_submatrix_unsafe(&s, A, dstix + 1, srcix, srcix - dstix - 1, 1);
    armas_x_submatrix_unsafe(&d, A, dstix, dstix + 1, 1, srcix - dstix - 1);
    armas_x_swap(&s, &d, conf);
    // S3 -- D3
    armas_x_submatrix_unsafe(&s, A, 0, srcix, dstix, 1);
    armas_x_submatrix_unsafe(&d, A, 0, dstix, dstix, 1);
    armas_x_swap(&s, &d, conf);
    // swap P1 and P3
    p1 = armas_x_get_unsafe(A, srcix, srcix);
    p3 = armas_x_get_unsafe(A, dstix, dstix);
    armas_x_set_unsafe(A, srcix, srcix, p3);
    armas_x_set_unsafe(A, dstix, dstix, p1);
}



/*
 * Swap rows of matrix
 */
static
void swap_rows(armas_x_dense_t * A, int src, int dst, armas_conf_t * conf)
{
    armas_x_dense_t r0, r1;
    if (src == dst || A->cols <= 0)
        return;
    if (src >= A->rows || dst >= A->rows)
        return;

    armas_x_submatrix(&r0, A, src, 0, 1, A->cols);
    armas_x_submatrix(&r1, A, dst, 0, 1, A->cols);
    armas_x_swap(&r0, &r1, (armas_conf_t *) 0);
}

/*
 * Swap columns of matrix
 */
static
void swap_cols(armas_x_dense_t * A, int src, int dst, armas_conf_t * conf)
{
    armas_x_dense_t r0, r1;
    if (src == dst || A->rows <= 0)
        return;
    if (src >= A->cols || dst >= A->cols)
        return;

    armas_x_submatrix(&r0, A, 0, src, A->rows, 1);
    armas_x_submatrix(&r1, A, 0, dst, A->rows, 1);
    armas_x_swap(&r0, &r1, (armas_conf_t *) 0);
}

/*
 * Apply row pivots forward or backward.
 */
static
void apply_row_pivots(armas_x_dense_t * A, armas_pivot_t * P,
                      int dir, armas_conf_t * conf)
{
    int k, n, nk;

    if (A->cols == 0)
        return;

    if (dir == ARMAS_PIVOT_FORWARD) {
        for (k = 0; k < P->npivots; k++) {
            n = P->indexes[k];
            if (n > 0 && n - 1 != k) {
                swap_rows(A, n - 1, k, conf);
            }
        }
    } else {
        // pivot index and row index may not coincide
        for (nk = A->rows - 1, k = P->npivots - 1; k >= 0; k--, nk--) {
            n = P->indexes[k];
            if (n > 0 && n - 1 != nk) {
                swap_rows(A, n - 1, nk, conf);
            }
        }
    }
}

/*
 * \brief Apply column pivots forward or backward.
 */
static
void apply_col_pivots(armas_x_dense_t * A, armas_pivot_t * P,
                      int dir, armas_conf_t * conf)
{
    int k, n, nk;

    if (A->rows == 0)
        return;

    if (dir == ARMAS_PIVOT_FORWARD) {
        for (k = 0; k < P->npivots; k++) {
            n = P->indexes[k];
            if (n > 0 && n - 1 != k) {
                swap_cols(A, n - 1, k, conf);
            }
        }
    } else {
        // pivot index and column index may not coincide (npivot < n(A))
        for (nk = A->cols - 1, k = P->npivots - 1; k >= 0; k--, nk--) {
            n = P->indexes[k];
            if (n > 0 && n - 1 != nk) {
                swap_cols(A, n - 1, nk, conf);
            }
        }
    }
}

/**
 * \brief Apply row pivots in P to matrix A forward or backward.
 *
 * \ingroup lapackaux
 */
int armas_x_pivot_rows(armas_x_dense_t * A, armas_pivot_t * P, int flags,
                       armas_conf_t * conf)
{
    if (A->rows < P->npivots)
        return -1;
    if (!conf)
        conf = armas_conf_default();

    apply_row_pivots(A, P, flags, conf);
    return 0;
}

/**
 * \brief Apply column pivots in P to matrix A forward or backward.
 *
 * \ingroup lapackaux
 */
int armas_x_pivot_cols(armas_x_dense_t * A, armas_pivot_t * P, int flags,
                       armas_conf_t * conf)
{
    if (A->cols < P->npivots)
        return -1;
    if (!conf)
        conf = armas_conf_default();

    apply_col_pivots(A, P, flags, conf);
    return 0;
}

/**
 * \brief Apply pivots to matrix
 *
 * Applies pivots defined in P to matrix A as indicated by control bits in flag parameter.
 * The control bits is combination of pivot direction indicator (ARMAS_PIVOT_FORWARD or 
 * ARMAS_PIVOT_BACKWARD) and pivot type flag. Possible mutually exclusive pivot types are 
 * ARMAS_PIVOT_ROWS, ARMAS_PIVOT_COLS, ARMAS_PIVOT_UPPER and ARMAS_PIVOT_LOWER. The UPPER
 * (LOWER) flags are for pivoting symmetric upper (lower) triangular matrices.
 * Default pivot direction is forward.
 *
 * Pivoting forward starts from the first row/column of matrix and first pivot index and 
 * moves down matrix rows/columns until all pivots are consumed. In forward pivoting
 * row/column indeces and pivot indeces coincide.
 *
 * Backward pivoting starts from the last row/column of matrix and the last pivot index
 * and moves up matrix rows/columns and pivot indexes until all pivots are consumed. In 
 * backward pivoting row/column indicies may not coincide with pivot indicies if pivot
 * array length is smaller than matrix row/column count.
 *
 * Pivot indexes are non-zero indexes to A rows or columns. Pivot index K is K-1 row or column
 * in the input matrix.
 *
 * \param[in,out] A
 *    On entry matrix to pivot. On exit pivoted matrix.
 * \param[in] P
 *    Pivots, non-zero index to A row or column. Length of P may be smaller than row/column
 *    of the input matrix.
 * \param[in] flags
 *    Pivot control flags
 */
int armas_x_pivot(armas_x_dense_t * A, armas_pivot_t * P, unsigned int flags,
                  armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    // get direction
    int k, kp, nk, dir = flags & 0x1;
    // mask out direction bit
    flags &= ~0x1;

    switch (flags) {
    case ARMAS_PIVOT_UPPER:
        if (A->rows < P->npivots) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        if (dir == ARMAS_PIVOT_FORWARD) {
            for (k = 0; k < P->npivots; k++) {
                kp = armas_pivot_get(P, k);
                if (kp - 1 == k)
                    continue;
                apply_pivot_upper(A, k, kp - 1, conf);
            }
        } else {
            // row/col index and pivot index may not coincide (npivots < n(A))
            for (nk = A->rows - 1, k = P->npivots - 1; k >= 0; k--, nk--) {
                kp = armas_pivot_get(P, k);
                if (kp - 1 == nk)
                    continue;
                apply_pivot_upper(A, kp - 1, nk, conf);
            }
        }
        break;

    case ARMAS_PIVOT_LOWER:
        if (A->rows < P->npivots) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        if (dir == ARMAS_PIVOT_FORWARD) {
            // pivot index k and row/col index coincide
            for (k = 0; k < P->npivots; k++) {
                kp = armas_pivot_get(P, k);
                if (kp - 1 == k)
                    continue;
                apply_pivot_lower(A, k, kp - 1, conf);
            }
        } else {
            // looping backwards from row count; pivot index need to compared
            // to current row number nk or pivot index k
            for (nk = A->rows - 1, k = P->npivots - 1; k >= 0; k--, nk--) {
                kp = armas_pivot_get(P, k);
                if (kp - 1 == nk)
                    continue;
                apply_pivot_lower(A, kp - 1, nk, conf);
            }
        }
        break;

    case ARMAS_PIVOT_COLS:
        if (A->cols < P->npivots) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        apply_col_pivots(A, P, dir, conf);
        break;

    case ARMAS_PIVOT_ROWS:
    default:
        if (A->rows < P->npivots) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        apply_row_pivots(A, P, dir, conf);
        break;
    }
    return 0;
}

/*
 * \brief Find index to largest absolute of vector.
 *
 * Find largest absolute value on a vector. Assumes A is vector. Returns non-zero
 * n and largest value at index n-1. On error returns zero.
 *
 * \ingroup lapackaux internal
 */
int armas_x_pivot_index(armas_x_dense_t * A, armas_conf_t * conf)
{
    return armas_x_iamax(A, conf) + 1;
}

/*
 * \brief Sort vector elements
 *
 * \param D [in]
 *      Vector to sort
 * \param P [in,out]
 *      Pivot table, on exit contains sorted order of vector
 * \param direction [in]
 *      Sort direction, >0 ascending, <0 descending
 * \return
 *      Non-zero if D not a vector, zero otherwise
 *
 * \ingroup lapackaux internal
 *
 * (This is maybe not so usefull at all, tricky to order based on this.)
 */
int armas_x_pivot_sort(armas_x_dense_t * D, armas_pivot_t * P, int direction)
{
    int k, j, pk, pj;
    DTYPE cval, tmpval;

    if (!armas_x_isvector(D)) {
        return -1;
    }
    // initialize the array
    for (k = 0; k < armas_x_size(D); k++) {
        armas_pivot_set(P, k, k + 1);
    }
    // simple indirect insertion sort
    for (k = 1; k < armas_x_size(D); k++) {
        pk = armas_pivot_get(P, k) - 1;
        cval = ABS(armas_x_get_at_unsafe(D, pk));
        for (j = k; j > 0; j--) {
            pj = armas_pivot_get(P, j - 1) - 1;
            tmpval = ABS(armas_x_get_at_unsafe(D, pj));
            if (direction < 0 && tmpval >= cval) {
                break;
            }
            if (direction > 0 && tmpval <= cval) {
                break;
            }
            armas_pivot_set(P, j, pj + 1);
        }
        armas_pivot_set(P, j, pk + 1);
    }
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_REQUIRES && ARMAS_PROVIDES */
