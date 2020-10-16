
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_luinverse) && defined(armas_luinverse_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external functions
#if defined(armas_inverse_trm) && defined(armas_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


extern
int unblk_inverse_upper(armas_dense_t * A, int flags, armas_conf_t * conf);

/*
 * general matrix inverse;
 * method: A = LU then solve  A*L = U^-1 for A^-1. This version uses fused
 * triangular upper matrix inverse within the loop.
*/
static
int unblk_inverse_fused(armas_dense_t * A, armas_dense_t * W,
                        armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a01, a11, a21, A22, l21;
    armas_dense_t AL, AR, A0, a1, A2;
    int err = 0;
    DTYPE a11val;

    EMPTY(A0);
    EMPTY(a11);
    EMPTY(ATL);
    EMPTY(AL);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_1x2(
        &AL, &AR, /**/ A, 0, ARMAS_PRIGHT);

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_1x2to1x3(
            &AL, &A0, &a1, &A2, /**/ A, 1, ARMAS_PLEFT);
        // ---------------------------------------------------------------------
        // inverse upper part;
        a11val = armas_get_unsafe(&a11, 0, 0);
        if (a11val == ZERO) {
            if (err == 0) {
                conf->error = ARMAS_ESINGULAR;
                err = -1;
            }
            goto next;
        }
        // fused upper triangular matrix inverse
        // 1. a11 = 1.0/a11
        a11val = ONE / a11val;
        armas_set_unsafe(&a11, 0, 0, a11val);
        // 2. a01 = -a11val*A00.-1*a01
        armas_mvsolve_trm(&a01, -a11val, &A00, ARMAS_UPPER, conf);

        // inverse A; l21 := a21; a21 = 0, we can compute full column a1 of A^-1.
        armas_make(&l21, a21.rows, a21.cols, a21.rows, armas_data(W));
        armas_copy(&l21, &a21, conf);
        armas_scale(&a21, ZERO, conf);

        // a1 := a1 - A2*l21
        armas_mvmult(ONE, &a1, -ONE, &A2, &l21, 0, conf);
        a11val = armas_get(&a11, 0, 0);
        // ---------------------------------------------------------------------
      next:
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_1x3to1x2(
            &AL, &AR, /**/ &A0, &a1, A, ARMAS_PLEFT);
    }
    return err;
}

static
int blk_inverse_fused(armas_dense_t * A, armas_dense_t * W, int lb,
                      armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A01, A11, A21, A22, LB, L11, L21;
    armas_dense_t AL, AR, A0, A1, A2, AB;
    int e, err = 0;

    EMPTY(A0);
    EMPTY(A11);
    EMPTY(ATL);
    EMPTY(AL);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_1x2(
        &AL, &AR, /**/ A, 0, ARMAS_PRIGHT);

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            __nil, &A11, __nil,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PTOPLEFT);
        mat_repartition_1x2to1x3(
            &AL, &A0, &A1, &A2, /**/ A, lb, ARMAS_PLEFT);
        // ---------------------------------------------------------------------
        // fused inverse upper triangular;
        // A11 := A11^-1
        if ((e = armas_inverse_trm(&A11, ARMAS_UPPER, conf)) < 0 && err == 0)
            err = e;
        // A01 := A01*A11
        armas_mult_trm(&A01, ONE, &A11, ARMAS_RIGHT | ARMAS_UPPER, conf);
        // A01 := -A00^-1*A01
        armas_solve_trm(&A01, -ONE, &A00, ARMAS_LEFT | ARMAS_UPPER, conf);

        // inverse A; copy A11,A21 to workspace
        mat_merge2x1(&AB, &A11, &A21);
        armas_make(&LB, AB.rows, A11.cols, AB.rows, armas_data(W));
        armas_mcopy(&LB, &AB, 0, conf);
        mat_partition_2x1(&L11, &L21, &LB, A11.rows, ARMAS_PTOP);
        // zero strictly lower triangular part of A11 and all of A21
        armas_mscale(&A11, ZERO, ARMAS_LOWER | ARMAS_UNIT, conf);
        armas_mscale(&A21, ZERO, 0, conf);

        // A1 := A1 - A2*L21
        armas_mult(ONE, &A1, -ONE, &A2, &L21, 0, conf);
        // A1 := A1*L11.-1
        armas_solve_trm(&A1, ONE, &L11,
                          ARMAS_LOWER | ARMAS_UNIT | ARMAS_RIGHT, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_1x3to1x2(
            &AL, &AR, /**/ &A0, &A1, A, ARMAS_PLEFT);
    }
    return err;
}

/**
 * \brief Computes the inverse of a general NxN matrix.
 *
 * \param[in,out] A
 *      On entry, the LU factorization of matrix A = LU. On exit the inverse of A
 * \param[out] W
 *      Workspace 
 * \param[in]  P
 *      Rows pivots of LU factorization
 * \param[in]  conf
 *      Configuration block
 *
 * \retval  0 Succes
 * \retval -1 Error, error code set in conf.error
 *
 */
int armas_luinverse(armas_dense_t * A,
                      const armas_pivot_t * P, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (!conf)
        conf = armas_conf_default();

    if (armas_luinverse_w(A, P, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_luinverse_w(A, P, wbs, conf);
    armas_wrelease(&wb);
    return err;
}


int armas_luinverse_w(armas_dense_t * A,
                        const armas_pivot_t * P,
                        armas_wbuf_t * wb, armas_conf_t * conf)
{
    int lb, err = 0;
    size_t wsmin, wsz;
    armas_dense_t Wt;
    armas_env_t *env;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->rows > env->lb)
            wb->bytes = A->rows * env->lb * sizeof(DTYPE);
        else
            wb->bytes = A->rows * sizeof(DTYPE);
        return 0;
    }

    if (A->rows != A->cols ||
        (P != ARMAS_NOPIVOT && armas_pivot_size(P) != A->rows)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    wsmin = A->rows * sizeof(DTYPE);
    wsz = armas_wbytes(wb);
    if (wsz < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    lb = env->lb;
    wsz /= sizeof(DTYPE);
    if (A->rows > lb) {
        if (wsz < lb * A->rows) {
            lb = (wsz / A->rows) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    armas_make(&Wt, wsz, 1, wsz, (DTYPE *) armas_wptr(wb));
    wsz = armas_wpos(wb);

    if (lb == 0 || A->rows <= lb) {
        err = unblk_inverse_fused(A, &Wt, conf);
    } else {
        err = blk_inverse_fused(A, &Wt, lb, conf);
    }

    if (err == 0 && P) {
        // apply col pivots ie. compute A := A*P
        armas_pivot(A, P, ARMAS_PIVOT_COLS | ARMAS_PIVOT_BACKWARD, conf);
    }
    armas_wsetpos(wb, wsz);
    return err;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
