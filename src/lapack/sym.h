
// Copyright (c) Harri Rautila, 2016

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_LAPACK_SYM_H
#define __ARMAS_LAPACK_SYM_H

/* 
 * Common definitions for symmetric algorithms: cholesky, bk, ldl
 */

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 *
 * UPPER triangular; moving from bottom-right to top-left (pivot root at the 
 * bottom-right corner of the block).
 *
 *    d x D3 x  x  x  S3 x |
 *      d D3 x  x  x  S3 x |
 *        P3 D2 D2 D2 P2 x |  -- dstinx
 *           d  x  x  S2 x |
 *              d  x  S2 x |
 *                 d  S2 x |
 *                    P1 x |  -- srcinx
 *                       d |
 *    ----------------------
 *               (ABR)
 */
static inline
void __apply_bkpivot_upper(armas_x_dense_t *AR, int srcix, int dstix, armas_conf_t *conf)
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
    // S2 -- D2
    armas_x_submatrix_unsafe(&s, AR, dstix+1, srcix,   srcix-dstix-1, 1);
    armas_x_submatrix_unsafe(&d, AR, dstix,   dstix+1, 1, srcix-dstix-1);
    armas_x_swap(&s, &d, conf);
    // S3 -- D3
    armas_x_submatrix_unsafe(&s, AR, 0, srcix,  dstix, 1);
    armas_x_submatrix_unsafe(&d, AR, 0, dstix,  dstix, 1);
    armas_x_swap(&s, &d, conf);
    // swap P1 and P3
    p1 = armas_x_get_unsafe(AR, srcix, srcix);
    p3 = armas_x_get_unsafe(AR, dstix, dstix);
    armas_x_set_unsafe(AR, srcix, srcix, p3);
    armas_x_set_unsafe(AR, dstix, dstix, p1);
}

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 *
 * UPPER triangular; moving from top-left to bottom-right eg. pivot root at
 * top left corner (columns above pivot index are not moved)
 *
 *  ----------------------------
 *  | d x  x  x  x  x  x  x  x  
 *  |   P1 S2 S2 P2 S3 S3 S3 S3   srcix
 *  |      d  x  D2 x  x  x  x  
 *  |         x  D2 x  x  x  x    
 *  |            P3 D3 D3 D2 D2   dstix
 *  |               d  x  x  x  
 *  |                  d  x  x  
 *  |                     d  x    
 *  |                        d 
 *
 */
static inline
void __apply_bkpivot_upper_top(armas_x_dense_t *AR, int srcix, int dstix, armas_conf_t *conf)
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
    // S2 -- D2
    armas_x_submatrix_unsafe(&s, AR, srcix,   srcix+1, 1, dstix-srcix-1);
    armas_x_submatrix_unsafe(&d, AR, srcix+1, dstix,   dstix-srcix-1, 1);
    armas_x_swap(&s, &d, conf);
    // S3 -- D3
    armas_x_submatrix_unsafe(&s, AR, srcix, dstix+1, 1, AR->cols-dstix-1);
    armas_x_submatrix_unsafe(&d, AR, dstix, dstix+1, 1, AR->cols-dstix-1);
    armas_x_swap(&s, &d, conf);
    // swap P1 and P3
    p1 = armas_x_get_unsafe(AR, srcix, srcix);
    p3 = armas_x_get_unsafe(AR, dstix, dstix);
    armas_x_set_unsafe(AR, srcix, srcix, p3);
    armas_x_set_unsafe(AR, dstix, dstix, p1);
}

/*
 * Apply diagonal pivot (row and column swapped) to symmetric LOWER triangular matrix blocks.
 * This is a partial swap; rows left srcix are not touched. (Pivot root is at the 
 * top-left corner of the block.)
 *
 *    -----------------------
 *    | d 
 *    | x P1 x  x  x  P2     -- current row/col 'srcix'
 *    | x S2 d  x  x  x
 *    | x S2 x  d  x  x
 *    | x S2 x  x  d  x
 *    | x P2 D2 D2 D2 P3     -- swap with row/col 'dstix'
 *    | x S3 x  x  x  D3 d
 *    | x S3 x  x  x  D3 x d
 *         (AR)
 */
static inline
void __apply_bkpivot_lower(armas_x_dense_t *AR, int srcix, int dstix, armas_conf_t *conf)
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
    // S2 -- D2
    armas_x_submatrix_unsafe(&s, AR, srcix+1, srcix,   dstix-srcix-1, 1);
    armas_x_submatrix_unsafe(&d, AR, dstix,   srcix+1, 1, dstix-srcix-1);
    armas_x_swap(&s, &d, conf);
    // S3 -- D3
    armas_x_submatrix_unsafe(&s, AR, dstix+1, srcix,  AR->rows-dstix-1, 1);
    armas_x_submatrix_unsafe(&d, AR, dstix+1, dstix,  AR->rows-dstix-1, 1);
    armas_x_swap(&s, &d, conf);
    // swap P1 and P3
    p1 = armas_x_get_unsafe(AR, srcix, srcix);
    p3 = armas_x_get_unsafe(AR, dstix, dstix);
    armas_x_set_unsafe(AR, srcix, srcix, p3);
    armas_x_set_unsafe(AR, dstix, dstix, p1);
}

/*
 * Apply diagonal pivot (row and column swapped) to symmetric LOWER triangular matrix blocks.
 * This is a partial swap; Pivot root (P1) is assumed to be at the bottom-right corner of 
 * the block. 
 *
 *
 *     d                      |
 *     x  d                   |
 *     D3 D3 P3               |  - dst
 *     x  x  D2 d             |
 *     x  x  D2 x  d          |
 *     x  x  D2 x  x  d       |
 *     S3 S3 P2 S2 S2 S2 P1   | - src
 *     x  x  x  x  x  x  x  d |
 *     ------------------------
 *
 */
static inline
void __apply_bkpivot_lower_bottom(armas_x_dense_t *AR, int srcix, int dstix, armas_conf_t *conf)
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
    // S2 -- D2
    armas_x_submatrix_unsafe(&s, AR, srcix,   dstix+1, 1, srcix-dstix-1);
    armas_x_submatrix_unsafe(&d, AR, dstix+1, dstix,   srcix-dstix-1, 1);
    armas_x_swap(&s, &d, conf);
    // S3 -- D3
    armas_x_submatrix_unsafe(&s, AR, srcix, 0, 1,  dstix);
    armas_x_submatrix_unsafe(&d, AR, dstix, 0, 1,  dstix);
    armas_x_swap(&s, &d, conf);
    // swap P1 and P3
    p1 = armas_x_get_unsafe(AR, srcix, srcix);
    p3 = armas_x_get_unsafe(AR, dstix, dstix);
    armas_x_set_unsafe(AR, srcix, srcix, p3);
    armas_x_set_unsafe(AR, dstix, dstix, p1);
}


/*
 * Swap rows of matrix
 */
static inline
void __swap_rows(armas_x_dense_t *A, int src, int dst, armas_conf_t *conf)
{
    armas_x_dense_t r0, r1;
    if (src == dst || A->cols <= 0)
        return;
    if ((unsigned int)src >= A->rows || (unsigned int)dst >= A->rows)
        return;

    armas_x_submatrix(&r0, A, src, 0, 1, A->cols);
    armas_x_submatrix(&r1, A, dst, 0, 1, A->cols);
    armas_x_swap(&r0, &r1, (armas_conf_t *)0);
}

/*
 * Swap columns of matrix
 */
static inline
void __swap_cols(armas_x_dense_t *A, int src, int dst, armas_conf_t *conf)
{
    armas_x_dense_t r0, r1;
    if (src == dst || A->rows <= 0)
        return;
    if (src >= A->cols || dst >= A->cols)
        return;
    
    armas_x_submatrix(&r0, A, 0, src, A->rows, 1);
    armas_x_submatrix(&r1, A, 0, dst, A->rows, 1);
    armas_x_swap(&r0, &r1, (armas_conf_t *)0);
}


#endif // __ARMAS_LAPACK_SYM_H

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
