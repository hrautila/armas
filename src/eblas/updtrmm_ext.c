
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Triangular/trapezoidal matrix rank update

//! \cond
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_update_trm) && defined(__update_trm_blk)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_kernel)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

/*
 * update diagonal block
 *
 *  l00           a00 a01   b00 b01 b02    u00 u01 u02
 *  l10 l11       a10 a11   b10 b11 b12        u11 u12
 *  l20 l21 l22   a20 a21                          u22
 *
 */
static
void ext_update_trm_diagonal(
    DTYPE beta,
    armas_x_dense_t *C,
    armas_x_dense_t *dC,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags)
{
    register int i, flg = 0;
    armas_x_dense_t A0, B0, c0, a0, b0, d0;

    if (flags & ARMAS_UPPER) {
        // index by row
        flg = flags & ARMAS_TRANSB ? 0 : ARMAS_TRANS;
        int M = min(C->rows, C->cols);
        for (i = 0; i < M; ++i) {
            switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
            case ARMAS_TRANSA|ARMAS_TRANSB:
                armas_x_column_unsafe(&a0, A, i);
                armas_x_submatrix_unsafe(&B0, B, i, 0, B->rows-i, B->cols);
                break;
            case ARMAS_TRANSA:
                armas_x_column_unsafe(&a0, A, i);
                armas_x_submatrix_unsafe(&B0, B, 0, i, B->rows, B->cols-i);
                break;
            case ARMAS_TRANSB:
                armas_x_row_unsafe(&a0, A, i);
                armas_x_submatrix_unsafe(&B0, B, i, 0, B->rows-i, B->cols);
                break;
            default:
                armas_x_row_unsafe(&a0, A, i);
                armas_x_submatrix_unsafe(&B0, B, 0, i, B->rows, B->cols-i);
                break;
            }
            armas_x_submatrix_unsafe(&c0, C, i, i, 1, C->cols-i);
            armas_x_submatrix_unsafe(&d0, dC, i, i, 1, C->cols-i);
            armas_x_scale_unsafe(&d0, __ZERO);
            armas_x_ext_mvmult_dx_unsafe(beta, &C0, &d0, alpha, &B0, &a0, flg);
            armas_x_ext_merge(&c0, &c0, &d0);
        }
    } else {
        // index by column
        int N = min(C->cols, C->rows);
        flg = flags & ARMAS_TRANSA ? ARMAS_TRANS : 0;
        for (i = 0; i < N; ++i) {
            switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
            case ARMAS_TRANSA|ARMAS_TRANSB:
                armas_x_row_unsafe(&b0, B, i);
                armas_x_submatrix_unsafe(&A0, A, 0, i, A->rows, A->cols-i);
                break;
            case ARMAS_TRANSA:
                armas_x_column_unsafe(&b0, B, i);
                armas_x_submatrix_unsafe(&A0, A, 0, i, A->rows, A->cols-i);
                break;
            case ARMAS_TRANSB:
                armas_x_row_unsafe(&b0, B, i);
                armas_x_submatrix_unsafe(&A0, A, i, 0, A->rows-i, A->cols);
                break;
            default:
                armas_x_column_unsafe(&b0, B, i);
                armas_x_submatrix_unsafe(&A0, A, i, 0, A->rows-i, A->cols);
                break;
            }
            armas_x_submatrix_unsafe(&c0, C, i, i, C->rows-i, 1);
            armas_x_submatrix_unsafe(&d0, dC, i, i, C->rows-i, 1);
            armas_x_scale_unsafe(&d0, __ZERO);
            armas_x_mvmult_unsafe(beta, &c0, &dc, alpha, &A0, &b0, flg);
            armas_x_ext_merge(&c0, &c0, &d0);
        }
    }
}



/*
 * Generic triangular matrix update:
 *      C = beta*op(C) + alpha*A*B
 *      C = beta*op(C) + alpha*A*B.T
 *      C = beta*op(C) + alpha*A.T*B
 *      C = beta*op(C) + alpha*A.T*B.T
 */
static
void update_trm_ext_blk(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    cache_t *cache)
{
    register int i, nI, N, M;
    armas_x_dense_t Cd, Ad, Bd, Dd;
    int NB = cache->NB;

    if ((flags & ARMAS_UPPER) != 0) {
        // by rows; M is the last row; L-S is column count; implicitely S == R
        M = min(C->rows, C->cols);
        for (i = 0; i < M; i += NB) {
            nI = M - i < NB ? M - i : NB;

            //printf("i=%dm nI=%d, L-i=%d, L-i-nI=%d\n", i, nI, L-i, L-i-nI);
            switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
            case ARMAS_TRANSA|ARMAS_TRANSB:
                armas_x_submatrix_unsafe(&Ad, A, 0, i, A->rows, nI);
                armas_x_submatrix_unsafe(&Bd, B, i, 0, nI, B->cols);
                break;
            case ARMAS_TRANSA:
                armas_x_submatrix_unsafe(&Ad, A, 0, i, A->rows, nI);
                armas_x_submatrix_unsafe(&Bd, B, 0, i, B->rows, nI);
                break;
            case ARMAS_TRANSB:
                armas_x_submatrix_unsafe(&Ad, A, i, 0, nI, A->cols);
                armas_x_submatrix_unsafe(&Bd, B, i, 0, nI, B->cols);
                break;
            default:
                armas_x_submatrix_unsafe(&Ad, A, i, 0, nI, A->cols);
                armas_x_submatrix_unsafe(&Bd, B, 0, i, B->rows, nI);
                break;
            }
            armas_x_submatrix_unsafe(&Cd, C, i,  i, nI, nI);
            armas_x_submatrix_unsafe(&Cd, C, i,  i, nI, nI);
            ext_update_trm_diagonal(beta, &Cd, alpha, &Ad, &Bd, flags, cache);

            // 2. update right of the diagonal block (rectangle, nI rows)
            switch (flags & ARMAS_TRANSB) {
            case ARMAS_TRANSB:
                armas_x_submatrix_unsafe(&Bd, B, i+nI, 0, B->rows-i-nI, B->cols);
                break;
            default:
                armas_x_submatrix_unsafe(&Bd, B, 0, i+nI, B->rows, B->cols-i-nI);
                break;
            }
            armas_x_submatrix_unsafe(&Cd, C, i, i+nI, nI, C->cols-i-nI);
            armas_x_ext_mult_kernel(beta, &Cd, alpha, &Ad, &Bd, flags, cache);
        }
        return;
    }
    // lower: by columns; N is the last column,
    N = min(C->rows, C->cols);
    for (i = 0; i < N; i += NB) {
        nI = N - i < NB ? N - i : NB;

        // 1. update on diagonal (square block)
        switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
        case ARMAS_TRANSA|ARMAS_TRANSB:
            armas_x_submatrix_unsafe(&Ad, A, 0, i, A->rows, nI);
            armas_x_submatrix_unsafe(&Bd, B, i, 0, nI, B->cols);
            break;
        case ARMAS_TRANSA:
            armas_x_submatrix_unsafe(&Ad, A, 0, i, A->rows, nI);
            armas_x_submatrix_unsafe(&Bd, B, 0, i, B->rows, nI);
            break;
        case ARMAS_TRANSB:
            armas_x_submatrix_unsafe(&Ad, A, i, 0, nI, A->cols);
            armas_x_submatrix_unsafe(&Bd, B, i, 0, nI, B->cols);
            break;
        default:
            armas_x_submatrix_unsafe(&Ad, A, i, 0, nI, A->cols);
            armas_x_submatrix_unsafe(&Bd, B, 0, i, B->rows, nI);
            break;
        }
        armas_x_submatrix_unsafe(&Cd, C, i,  i, nI, nI);
        ext_update_trm_diagonal(beta, &Cd, &Dd, alpha, &Ad, &Bd, flags, cache);

        // 2. update block below the diagonal block (rectangle, nI columns)
        switch (flags & ARMAS_TRANSA) {
        case ARMAS_TRANSA:
            armas_x_submatrix_unsafe(&Ad, A, 0, i+nI, A->rows, A->cols-i-nI);
            break;
        default:
            armas_x_submatrix_unsafe(&Ad, A, i+nI, 0, A->rows-i-nI, A->cols);
            break;
        }
        armas_x_submatrix_unsafe(&Cd, C, i+nI, i, C->rows-i-nI, nI);
        armas_x_ext_mult_kernel(beta, &Cd, alpha, &Ad, &Bd, flags, cache);
    }
}

/**
 * @brief Triangular or trapezoidial matrix rank-k update
 *
 * Computes
 *   - \f$ C = beta \times C + alpha \times A B \f$
 *   - \f$ C = beta \times C + alpha \times A^T B  \f$ if *ARMAS_TRANSA* set
 *   - \f$ C = beta \times C + alpha \times A B^T  \f$ if *ARMAS_TRANSB* set
 *   - \f$ C = beta \times C + alpha \times A^T B^T \f$ if *ARMAS_TRANSA* and *ARMAS_TRANSB* set
 *
 * Matrix C is upper (lower) triangular or trapezoidial if flag bit
 * *ARMAS_UPPER* (*ARMAS_LOWER*) is set. If matrix is upper (lower) then
 * the strictly lower (upper) part is not referenced.
 *
 * @param[in] beta scalar constant
 * @param[in,out] C triangular/trapezoidial result matrix
 * @param[in] alpha scalar constant
 * @param[in] A first operand matrix
 * @param[in] B second operand matrix
 * @param[in] flags matrix operand indicator flags
 * @param[in,out] conf environment configuration
 *
 * @retval 0  Operation succeeded
 * @retval <0 Failed, conf.error set to actual error code.
 *
 * @ingroup blas3
 */
int armas_x_ext_update_trm(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (armas_x_size(C) == 0 || armas_x_size(A) == 0 || armas_x_size(B) == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
    case ARMAS_TRANSA|ARMAS_TRANSB:
        ok = A->rows == B->cols && C->cols == B->rows && C->rows == A->cols;
        break;
    case ARMAS_TRANSA:
        ok = A->rows == B->rows && C->cols == B->cols && C->rows == A->cols;
        break;
    case ARMAS_TRANSB:
        ok = A->cols == B->cols && C->cols == B->rows && C->rows == A->rows;
        break;
    default:
        ok = A->cols == B->rows && C->cols == B->cols && C->rows == A->rows;
        break;
    }
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;
    if (armas_cbuf_select(&cbuf, conf) < 0) {
        conf->error = ARMAS_EMEMORY;
        return -1;
    }

    cache_t cache;
    armas_cache_setup(&cache, &cbuf, 3, sizeof(DTYPE));

    update_trm_ext_blk(beta, C, alpha, A, B, flags, &cache);
    armas_cbuf_release(&cbuf);
    return 0;
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
