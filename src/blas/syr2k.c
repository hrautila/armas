
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Symmetric matrix rank-2k update

//! \cond
#include <stdio.h>
#include <string.h>
//! \endcond
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_update2_sym)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_update_trm)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
//! \endcond

static
void update_syr2k_unb(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    armas_x_dense_t ATL, ABR;
    armas_x_dense_t A00, a01, a10, a11, A22;
    armas_x_dense_t XT, XB, X0, x1, X2;
    armas_x_dense_t YT, YB, Y0, y1, Y2;
    DTYPE ak;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &XT,
        &XB, /* */, X, 0, ARMAS_PTOP);
    mat_partition_2x1(
        &YT,
        &YB, /* */, Y, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00,  &a01, __nil,
            &a10,  &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &XT, &X0, &x1, &X2, /**/ X, 1, ARMAS_PBOTTOM);
        mat_repartition_2x1to3x1(
            &YT, &Y0, &y1, &Y2, /**/ Y, 1, ARMAS_PBOTTOM);
        // -------------------------------------------------------------------
        ak = ZERO;
        armas_x_adot_unsafe(&ak, 2.0*alpha, &x1, &y1);
        ak += armas_get_unsafe(&a11, 0, 0) * beta;
        armas_set_unsafe(&a11, 0, 0, ak);
        if (flags & ARMAS_UPPER) {
            armas_x_mvmult_unsafe(beta, &a01, alpha, &Y0, &x1, 0);
            armas_x_mvmult_unsafe(ONE, &a01, alpha, &X0, &y1, 0);
        } else {
            armas_x_mvmult_unsafe(beta, &a10, alpha, &Y0, &x1, 0);
            armas_x_mvmult_unsafe(ONE, &a10, alpha, &X0, &y1, 0);
        }
        // -------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &XT, &XB, /**/ &X0, &x1, X, ARMAS_PBOTTOM);
        mat_continue_3x1to2x1(
            &YT, &YB, /**/ &Y0, &y1, Y, ARMAS_PBOTTOM);

    }
}

static
void update_syr2k_trans_unb(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags)
{
    armas_x_dense_t ATL, ABR;
    armas_x_dense_t A00, a01, a10, a11, A22;
    armas_x_dense_t XL, XR, X0, x1, X2;
    armas_x_dense_t YL, YR, Y0, y1, Y2;
    DTYPE ak;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &XL, &XR, /* */, X, 0, ARMAS_PLEFT);
    mat_partition_1x2(
        &YL, &YR, /* */, Y, 0, ARMAS_PLEFT);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00,  &a01, __nil,
            &a10,  &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &XL, &X0, &x1, &X2, /**/ X, 1, ARMAS_PRIGHT);
        mat_repartition_2x1to3x1(
            &YL, &Y0, &y1, &Y2, /**/ Y, 1, ARMAS_PRIGHT);
        // -------------------------------------------------------------------
        ak = ZERO;
        armas_x_adot_unsafe(&ak, 2.0*alpha, &x1, &y1);
        ak += armas_get_unsafe(&a11, 0, 0) * beta;
        armas_set_unsafe(&a11, 0, 0, ak);
        if (flags & ARMAS_UPPER) {
            armas_x_mvmult_unsafe(beta, &a01, alpha, &Y0, &x1, ARMAS_TRANS);
            armas_x_mvmult_unsafe(ONE, &a01, alpha, &X0, &y1, ARMAS_TRANS);
        } else {
            armas_x_mvmult_unsafe(beta, &a10, alpha, &Y0, &x1, ARMAS_TRANS);
            armas_x_mvmult_unsafe(ONE, &a10, alpha, &X0, &y1, ARMAS_TRANS);
        }
        // -------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &XL, &XR, /**/ &X0, &x1, X, ARMAS_PRIGHT);
        mat_continue_3x1to2x1(
            &YL, &YR, /**/ &Y0, &y1, Y, ARMAS_PRIGHT);
    }
}

static
void update_syr2k_trans_recursive(
    DTYPE beta,
    armas_x_dense_t *A,
    DTYPE alpha,
    const armas_x_dense_t *X,
    const armas_x_dense_t *Y,
    int flags,
    int minblock,
    cache_t *cache)
{
    armas_x_dense_t ATL, ATR, ABL, ABR;
    armas_x_dense_t XL, XR, YL, YR;

    if (A->rows < minblock) {
        6update_syr2k_trans_unb(beta, A, alpha, X, Y, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->rows/2, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &XL, &XR, /* */, X, A->rows/2, ARMAS_PLEFT);
    mat_partition_1x2(
        &YL, &YR, /* */, Y, A->rows/2, ARMAS_PLEFT);

    update_syr2k_trans_recursive(beta, &ATL, alpha, &XL, &YL, flags, minblock);
    if (flags & ARMAS_UPPER) {
        armas_x_mult_kernel(beta, &ATR, alpha, &XL, &YR, ARMAS_TRANSA, cache);
        armas_x_mult_kernel_nc(&ATR, alpha, &YL, &XR, ARMAS_TRANSA, cache);
    } else {
        armas_x_mult_kernel(beta, &ABL, alpha, &XR, &YL, ARMAS_TRANSA, cache);
        armas_x_mult_kernel_nc(&ABL, alpha, &YR, &XL, ARMAS_TRANSA, cache);
    }
    update_syr2k_trans_recursive(beta, &ABR, alpha, &XR, &YR, flags, minblock);
}

/**
 * @brief Symmetric matrix rank-2k update
 *
 * Computes
 * - \f$ C = beta \times C + alpha \times A B^T + alpha \times B A^T \f$
 * - \f$ C = beta \times C + alpha \times A^T B + alpha \times B^T A \f$ if *ARMAS_TRANSA* set
 *
 * Matrix C has elements stored in the  upper (lower) triangular part
 * if flag bit *ARMAS_UPPER* (*ARMAS_LOWER*) is set.
 * If matrix is upper (lower) then the strictly lower (upper) part is not referenced.
 *
 * @param[in] beta scalar constant
 * @param[in,out] C result matrix
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
int armas_x_update2_sym(
    DTYPE beta,
    armas_x_dense_t *C,
    DTYPE alpha,
    const armas_x_dense_t *A,
    const armas_x_dense_t *B,
    int flags,
    armas_conf_t *conf)
{
    if (armas_x_size(C) == 0 || armas_x_size(A) == 0 || armas_x_size(B) == 0)
        return 0;

    if (!conf)
        conf = armas_conf_default();

    int ok = A->rows == B->rows && A->cols == B->cols && C->rows == C->cols;
    switch (flags & ARMAS_TRANS) {
    case ARMAS_TRANS:
        ok = ok && C->rows == A->cols;
        break;
    default:
        ok = ok && C->rows == A->rows;
        break;
    }
    if (!ok) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    int uflags = flags & (ARMAS_UPPER | ARMAS_LOWER);
    // do it twice with triangular update
    if (flags & ARMAS_TRANS) {
        if (armas_x_update_trm(beta, C,
                               alpha, A, B, uflags|ARMAS_TRANSA, conf) < 0)
            return -1;
        return armas_x_update_trm(ONE, C,
                                  alpha, B, A, uflags | ARMAS_TRANSA, conf);
    }
    if (armas_x_update_trm(beta, C
                           , alpha, A, B, uflags | ARMAS_TRANSB, conf) < 0)
        return -1;
    return armas_x_update_trm(ONE, C, alpha, B, A, uflags | ARMAS_TRANSB, conf);
}
#else
#warning "Missing defines; no code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
