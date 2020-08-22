
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Matrix-matrix multiplication with symmetric matrix

//! \cond
#include <stdio.h>
#include <stdlib.h>
//! \endcond

#include "dtype.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_mult_sym) 
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_kernel_inner) && defined(armas_x_mult_kernel_nc)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "matcpy.h"
#ifdef CONFIG_ACCELERATORS
#include "accel.h"
#endif /* CONFIG_ACCELERATORS */
//! \endcond

// C += A*B; A is the diagonal block
static
void mult_symm_diag(
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    cache_t *cache)
{
    armas_x_dense_t Acpy, Bcpy;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    /*
     * upper/lower part of source A untouchable, copy triangular block and fill
     * lower/upper part
     */
    if (flags & ARMAS_RIGHT) {
        armas_x_make(&Acpy, A->rows, A->cols, cache->ab_step, cache->Acpy);
        if (flags & ARMAS_LOWER) {
            CPTRIL_UFILL(&Acpy, A, A->rows, A->cols, unit);
        } else {
            CPTRIU_LFILL(&Acpy, A, A->rows, A->cols, unit);
        }
        if ((flags & ARMAS_TRANSB) != 0) {
            armas_x_make(&Bcpy, B->rows, B->cols, cache->ab_step, cache->Bcpy);
            CPBLK(&Bcpy, B, B->rows, B->cols, flags);
        } else {
            armas_x_make(&Bcpy, B->cols, B->rows, cache->ab_step, cache->Bcpy);
            CPBLK_TRANS(&Bcpy, B, B->rows, B->cols, flags);
        }
        armas_x_mult_kernel_inner(C, alpha, &Bcpy, &Acpy, cache->rb);
    } else {
        armas_x_make(&Acpy, A->rows, A->cols, cache->ab_step, cache->Acpy);
        if (flags & ARMAS_LOWER) {
            CPTRIL_UFILL(&Acpy, A, A->rows, A->cols, unit);
        } else {
            CPTRIU_LFILL(&Acpy, A, A->rows, A->cols, unit);
        }
        if ((flags & ARMAS_TRANSB) != 0) {
            armas_x_make(&Bcpy, B->cols, B->rows, cache->ab_step, cache->Bcpy);
            CPBLK_TRANS(&Bcpy, B, B->rows, B->cols, flags);
        } else {
            armas_x_make(&Bcpy, B->rows, B->cols, cache->ab_step, cache->Bcpy);
            CPBLK(&Bcpy, B, B->rows, B->cols, flags);
        }
        armas_x_mult_kernel_inner(C, alpha, &Acpy, &Bcpy, cache->rb);
    }
}

static
void armas_x_mult_symm_left(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    int P,
    cache_t *mcache)
{
    int i, j, nI, nJ, flags1, flags2, r, c, nr, nc;
    int mb, nb;
    armas_x_dense_t A0, B0, C0;

    if (alpha == 0.0) {
        if (beta != 1.0) {
            armas_x_scale_unsafe(C, beta);
        }
        return;
    }
    flags1 = 0;
    flags2 = 0;

    /*
     * P indexes column/row of A to column/row corresponding C, B matrix.
     * On single threaded case P == 0. 
     *
     *   upper:                lower:
     *        . A0 .   B0           .  .  .  B0
     *   C0 = . A1 A2  B1      C0 = A0 A1 .  B1
     *        . .  .   B2           .  A2 .  B2
     *
     * C = C + A0*B0 + A1*B1 + A2*B2
     */
    flags1 |= (flags & ARMAS_UPPER) != 0 ? ARMAS_TRANSA : 0;
    flags2 |= (flags & ARMAS_UPPER) != 0 ? 0 : ARMAS_TRANSA;
    if ((flags & ARMAS_TRANSB) != 0) {
        flags1 |= ARMAS_TRANSB;
        flags2 |= ARMAS_TRANSB;
    }
    mb = C->rows < mcache->MB ? C->rows : mcache->MB;
    nb = C->cols < mcache->NB ? C->cols : mcache->NB;
    for (i = 0; i < C->rows; i += mb) {
        nI = C->rows - i < mb ? C->rows - i : mb;

        // for all column of C, B ...
        for (j = 0; j < C->cols; j += nb) {
            nJ = C->cols - j < nb ? C->cols - j : nb;
            armas_x_submatrix_unsafe(&C0, C, i, j, nI, nJ);

            // block of C upper left at [i,j], lower right at [i+nI, j+nj]
            armas_x_scale_unsafe(&C0, beta);

            // off diagonal block in A; if UPPER then above [i,j]; if LOWER then
            // left of [i,j] above|left diagonal
            r  = (flags & ARMAS_UPPER) != 0 ? 0 : P+i;
            c  = (flags & ARMAS_UPPER) != 0 ? P+i : 0;
            nr = (flags & ARMAS_UPPER) != 0 ? P+i : nI;
            nc = (flags & ARMAS_UPPER) != 0 ? nI : P+i;
            armas_x_submatrix_unsafe(&A0, A, r, c, nr, nc);
            if ((flags & ARMAS_TRANSB) != 0) {
                armas_x_submatrix_unsafe(&B0, B, j, 0, nJ, P+i);
            } else {
                armas_x_submatrix_unsafe(&B0, B, 0, j, P+i, nJ);
            }

            armas_x_mult_kernel_nc(&C0, alpha, &A0, &B0, flags1, mcache);

            // on-diagonal block in A;
            armas_x_submatrix_unsafe(&A0, A, P+i, P+i, nI, nI);
            if ((flags & ARMAS_TRANSB) != 0) {
                armas_x_submatrix_unsafe(&B0, B, j, i, nJ, nI);
            } else {
                armas_x_submatrix_unsafe(&B0, B, i, j, nI, nJ);
            }
            mult_symm_diag(&C0, alpha, &A0, &B0, flags, mcache);

            // off-diagonal block in A; if UPPER then right of [i, i+nI];
            // if LOWER then below [i+nI, i]
            r  = P + ((flags & ARMAS_UPPER) != 0 ? i : i + nI);
            c  = P + ((flags & ARMAS_UPPER) != 0 ? i + nI : i);
            nr = (flags & ARMAS_UPPER) != 0 ? nI : A->cols - r;
            nc = (flags & ARMAS_UPPER) != 0 ? A->cols - c : nI;
            armas_x_submatrix_unsafe(&A0, A, r, c, nr, nc);
            if ((flags & ARMAS_TRANSB) != 0) {
                armas_x_submatrix_unsafe(&B0, B, j, i+nI, nJ, B->cols - i - nI);
            } else {
                armas_x_submatrix_unsafe(&B0, B, i+nI, j, B->rows - i - nI, nJ);
            }
            armas_x_mult_kernel_nc(&C0, alpha, &A0, &B0, flags2, mcache);
        }
    }
}


static
void armas_x_mult_symm_right(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    int P,
    cache_t *mcache)
{
    int flags1, flags2;
    register int nR, nC, ic, ir, r, c, nr, nc;
    armas_x_dense_t A0, B0, C0;

    if (alpha == 0.0) {
        if (beta != 1.0) {
            armas_x_scale_unsafe(C, beta);
        }
        return;
    }
    flags1 = 0;
    flags2 = 0;

    /*
     * P is row/column number accessing A matrix. [P, P+B->rows]
     *
     *   upper:                lower:
     *                 . A0 .                  .  .  .
     *   C0 = B0 B1 B2 . A1 A2   C0 = B0 B1 B2 A0 A1 .
     *                 . .  .                  .  A2 .
     *
     * update nR,nC block of C with stripe of B of nR rows
     * and stripe of A with nC columns.
     *
     * C = C + A0*B0 + A1*B1 + A2*B2
     */

    flags1 = (flags & ARMAS_TRANSB) != 0 ? ARMAS_TRANSA : 0;
    flags2 = (flags & ARMAS_TRANSB) != 0 ? ARMAS_TRANSA : 0;

    flags1 |= (flags & ARMAS_UPPER) != 0 ? 0 : ARMAS_TRANSB;
    flags2 |= (flags & ARMAS_UPPER) != 0 ? ARMAS_TRANSB : 0;

    int mb = C->rows < mcache->MB ? C->rows : mcache->MB;
    int nb = C->cols < mcache->NB ? C->cols : mcache->NB;
    for (ic = 0; ic < C->cols; ic += nb) {
        nC = C->cols - ic < nb ? C->cols - ic : nb;

        // for all rows of C, B ...
        for (ir = 0; ir < C->rows; ir += mb) {
            nR = C->rows - ir < mb ? C->rows - ir : mb;

            armas_x_submatrix_unsafe(&C0, C, ir, ic, nR, nC);
            armas_x_scale_unsafe(&C0, beta);

            // above|left diagonal
            r  = (flags & ARMAS_UPPER) != 0 ? 0 : P + ic;
            c  = (flags & ARMAS_UPPER) != 0 ? P + ic : 0;
            nr = (flags & ARMAS_UPPER) != 0 ? P + ic : nC;
            nc = (flags & ARMAS_UPPER) != 0 ? nC : P + ic;
            armas_x_submatrix_unsafe(&A0, A, r, c, nr, nc);
            if ((flags & ARMAS_TRANSB) != 0) {
                armas_x_submatrix_unsafe(&B0, B, 0, ir, P + ic, nR);
            } else {
                armas_x_submatrix_unsafe(&B0, B, ir, 0, nR, P + ic);
            }
            armas_x_mult_kernel_nc(&C0, alpha, &B0, &A0, flags1, mcache);

            // diagonal block
            armas_x_submatrix_unsafe(&A0, A, P+ic, P+ic, nC, nC);
            if ((flags & ARMAS_TRANSB) != 0) {
                armas_x_submatrix(&B0, B, ic, ir, nC, nR);
            } else {
                armas_x_submatrix(&B0, B, ir, ic, nR, nC);
            }
            mult_symm_diag(&C0, alpha, &A0, &B0, flags, mcache);

            // right|below of diagonal
            r  = (flags & ARMAS_UPPER) != 0 ? P + ic : P + ic + nC;
            c  = (flags & ARMAS_UPPER) != 0 ? P + ic + nC : P + ic;
            nr = (flags & ARMAS_UPPER) != 0 ? nC : A->cols - (P + ic + nC);
            nc = (flags & ARMAS_UPPER) != 0 ? A->cols - (P + ic + nC) : nC;
            armas_x_submatrix_unsafe(&A0, A, r, c, nr, nc);
            if ((flags & ARMAS_TRANSB) != 0) {
                armas_x_submatrix_unsafe(&B0, B, ic+nC, ir,
                                         B->rows - (P + ic + nC), nR);
            } else {
                armas_x_submatrix_unsafe(&B0, B, ir, ic+nC,
                                         nR, B->cols - (P + ic + nC));
            }
            armas_x_mult_kernel_nc(&C0, alpha, &B0, &A0, flags2, mcache);
        }
    }
}

/**
 * @brief Symmetric matrix-matrix multiplication
 *
 * If flag *ARMAS_LEFT* is set computes
 *   - \f$ C = alpha \times A B + beta \times C \f$
 *   - \f$ C = alpha \times A B^T + beta \times C   \f$ if *ARMAS_TRANSB* set
 *
 * If flag *ARMAS_RIGHT* is set computes
 *   - \f$ C = alpha \times B A + beta \times C    \f$
 *   - \f$ C = alpha \times B^T A + beta \times C  \f$ if *ARMAS_TRANSB* set
 *
 * Matrix A elements are stored on lower (upper) triangular part of the matrix
 * if flag bit *ARMAS_LOWER* (*ARMAS_UPPER*) is set.
 *
 * If option *ARMAS_OEXTPREC* is set in *conf.optflags* then computations
 * are executed in extended precision.
 *
 * @param[in] beta scalar constant
 * @param[in,out] C result matrix
 * @param[in] alpha scalar constant
 * @param[in] A symmetric matrix
 * @param[in] B second operand matrix
 * @param[in] flags matrix operand indicator flags
 * @param[in,out] conf environment configuration
 *
 * @retval   0  Operation succeeded
 * @retval < 0  Failed, conf.error set to actual error code.
 *
 * @ingroup blas3
 */
int armas_x_mult_sym(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    armas_conf_t *conf)
{
    int ok;

    if (C->rows == 0 || C->cols == 0)
        return 0;
    if (armas_x_size(A) == 0 || armas_x_size(B) == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    // check consistency
    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT|ARMAS_TRANSB)) {
    case ARMAS_RIGHT|ARMAS_TRANSB:
        ok = C->rows == B->cols
            && C->cols == A->cols
            && B->rows == A->rows
            && A->rows == A->cols;
        break;
    case ARMAS_RIGHT:
        ok = C->rows == B->rows
            && C->cols == A->cols
            && B->cols == A->rows
            && A->rows == A->cols;
        break;
    case ARMAS_LEFT|ARMAS_TRANSB:
    case ARMAS_TRANSB:
        ok = C->rows == A->rows
            && C->cols == B->rows
            && A->cols == B->cols
            && A->rows == A->cols;
        break;
    case ARMAS_LEFT:
    default:
        ok = C->rows == A->rows
            && C->cols == B->cols
            && A->cols == B->rows
            && A->rows == A->cols;
        break;
    }
    if (! ok) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }

    if (CONFIG_ACCELERATORS) {
        struct armas_ac_blas3 args;
        armas_ac_set_blas3_args(&args, beta, C, alpha, A, B, flags);
        int rc = armas_ac_dispatch(conf->accel, ARMAS_AC_SYMM, &args, conf);
        if (rc != -ARMAS_EIMP)
            return rc;
        /* fallthru to local version. */
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;

    if (armas_cbuf_select(&cbuf, conf) < 0) {
        conf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    cache_t cache;
    armas_env_t *env = armas_getenv();
    armas_cache_setup2(&cache, &cbuf, env->mb, env->nb, env->kb, sizeof(DTYPE));

    if (flags & ARMAS_RIGHT) {
        armas_x_mult_symm_right(beta, C, alpha, A, B, flags, 0, &cache);
    } else {
        armas_x_mult_symm_left(beta, C, alpha, A, B, flags, 0, &cache);
    }
    armas_cbuf_release(&cbuf);
    return 0;
}

void armas_x_mult_sym_unsafe(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    int K,
    cache_t *cache)
{
    if (flags & ARMAS_RIGHT) {
        armas_x_mult_symm_right(beta, C, alpha, A, B, flags, K, cache);
    } else {
        armas_x_mult_symm_left(beta, C, alpha, A, B, flags, K, cache);
    }
}

#else
#warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
