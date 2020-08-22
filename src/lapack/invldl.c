
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ldlinverse) && defined(armas_x_ldlinverse_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
#include <assert.h>
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

#include "sym.h"


/*
 * precondition:
 *      A = L*L.T && L = TriL(A)
 * invariant : 
 *      A*L = L.-T D.-1
 * pre-update : 
 *      A22*L22 = L22.-T*D2.-1
 * post-update:
 *      (a11 a21.T) ( I     ) = ( I l21.-T) (d1.-1   0  )
 *      (a21 A22  ) (l21 L22)   ( 0 l22.-T) (  0   D2.-1)
 *
 *      (1)  a11 + a21.T*l21 = d1.-1 --> a11 = d1.-1 - a21.T*L21
 *      (2)  a21 + A22*l21   = 0     --> a21 = -A22*L21
 *
 * post-update: (alternatively if invariant is A*L*D = L.-T
 *      (a11 a21.T) ( I     ) (d1  0 ) = ( I l21.-T)
 *      (a21 A22  ) (l21 L22) (0   D2)   ( 0 l22.-T)
 *
 *      (a11 a21.T) (    d1     0 ) = ( I l21.-T)
 *      (a21 A22  ) (l21*d1 L22*D2)   ( 0 l22.-T)
 *
 *      (1)  a11*d1 + a21.T*l21*d1 = I --> a11 = d1.-1 - a21.T*L21
 *      (2)  a21*d1 + A22*l21*d1   = 0 --> a21 = -A22*l21
 *
 */
static
int unblk_invldl_lower(armas_x_dense_t * A, armas_x_dense_t * W,
                       armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a21, A22, l21;
    int err = 0;
    DTYPE a11val, udot;

    EMPTY(ATL);
    EMPTY(a11);
    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------
        armas_x_make(&l21, a21.rows, 1, a21.rows, armas_x_data(W));
        armas_x_copy(&l21, &a21, conf);
        // a21 = - A22*l21
        armas_x_mvmult_sym(ZERO, &a21, -ONE, &A22, &l21,
                           ARMAS_LOWER | ARMAS_LEFT, conf);
        // a11 = a11 - a21.T*l21
        udot = armas_x_dot(&a21, &l21, conf);
        a11val = ONE / armas_x_get_unsafe(&a11, 0, 0);
        armas_x_set_unsafe(&a11, 0, 0, a11val - udot);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PTOPLEFT);
    }
    return err;
}


/*
 * precondition:
 *   A = L*L.T && L = TriL(A)
 * invariant : 
 *   A*L = L.-T D.-1
 * pre-update : 
 *   A22*L22 = L22.-T*D2.-1
 * post-update:
 *   (A11 A21.T) (L11    ) = (L11.-T L21.-T) (D1.-1   0  )
 *   (A21 A22  ) (L21 L22)   (   0   L22.-T) (  0   D2.-1)
 *
 *   (1)  A11*L11 + A21.T*L21 = L11.-T*D1.-1 -->
 *            A11 = L11.-T*D1.1*L11.-1 - A21.T*L21*L11.-1
 *   (2)  A21*L11 + A22*L21   = 0 -->
 *             A21 = (-A22*L21)*L11.-1
 */
static
int blk_invldl_lower(armas_x_dense_t * A, armas_x_dense_t * W, int lb,
                     armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A21, A22, L21, w2;
    int err = 0;

    EMPTY(ATL);
    EMPTY(A00);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, __nil,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PTOPLEFT);
        // ---------------------------------------------------------------------
        // workspaces;
        armas_x_make(&w2, lb, 1, lb, armas_x_data(W));
        armas_x_make(&L21, A21.rows, A11.cols, A21.rows, &armas_x_data(W)[lb]);
        armas_x_mcopy(&L21, &A21, 0, conf);

        // A21 = - A22*L21*L11.-1
        armas_x_mult_sym(ZERO, &A21, -ONE, &A22, &L21, ARMAS_LOWER, conf);
        armas_x_solve_trm(&A21, ONE, &A11,
                          ARMAS_LOWER | ARMAS_UNIT | ARMAS_RIGHT, conf);

        // A11*L11 + A21.T*L21 = L11.-T -> A11 = inv(A11) - A21.T*L21*L11.-1
        armas_x_solve_trm(&L21, ONE, &A11,
                          ARMAS_LOWER | ARMAS_UNIT | ARMAS_RIGHT, conf);
        if (unblk_invldl_lower(&A11, &w2, conf) < 0 && err == 0)
            err = -1;
        // rank update
        armas_x_update_trm(ONE, &A11, -ONE, &A21, &L21,
                           ARMAS_TRANSA | ARMAS_LOWER, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PTOPLEFT);
    }
    return err;
}


/*
 * precondition:
 *   A = U*D*U.T && U = TriU(A)
 * invariant : 
 *   A*U*D = U.-T 
 * pre-update : 
 *   A00*u00*d1 = U00.-T
 * post-update:
 *   (A00   a01) (U00 u01)(D0  0) = ( U00.-T  0)
 *   (a01.T a11) ( 0   I )( 0 d1)   ( u01.-t  I)
 *
 *   (A00   a01) (U00*D0  u01*d1) = (U00.-T 0)
 *   (a01.T a11) (   0      d1  ) = (u01.-T I)
 *
 *   (1) A00*u01*d1 + a01*d1   = 0  --> a01 = -A00*u01   (*)
 *   (2) a01.T*u01*d1 + a11*d1 = I  --> a11 = d1.-1 - a01.T*u01 (*)
 *
 */
static
int unblk_invldl_upper(armas_x_dense_t * A, armas_x_dense_t * W,
                         armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a01, a11, A22, u01;
    int err = 0;
    DTYPE a11val, udot;

    EMPTY(A00);
    EMPTY(a11);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    while (ABR.rows > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        armas_x_make(&u01, a01.rows, 1, a01.rows, armas_x_data(W));
        armas_x_copy(&u01, &a01, conf);

        armas_x_mvmult_sym(ZERO, &a01, -ONE, &A00, &u01, ARMAS_UPPER, conf);
        a11val = ONE / armas_x_get_unsafe(&a11, 0, 0);
        udot = armas_x_dot(&a01, &u01, conf);
        armas_x_set_unsafe(&a11, 0, 0, a11val - udot);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}

/*
 * precondition:
 *   A = U*D*U.T && U = TriU(A)
 * invariant : 
 *   A*U*D = U.-T
 * pre-update : 
 *   A00*U00*D0 = U00.-T
 * post-update:
 *   (A00   A01)(U00 U01)(D0  0) = (U00.-T   0   )
 *   (A01.T A11)( 0  U11)(0  D1)    U01.-T U11.-T)
 *
 *   (A00   A01)(U00*D0 U01*D1) = (U00.-T   0   )
 *   (A01.T A11)( 0     U11*D1)   (U01.-T U11.-T)
 *
 *   (1)  A00*U01*D1 + A01*U11*D1   = 0      -->  A01 = -A00*U01*U11.-1
 *   (2)  A01.T*U01*D1 + A11*U11*D1 = U11.-T -->
 *            A11 = U11.-T*D1.-1*U11.-1 - A01.T*U01*U11.-1 ->
 *            A11 = inv(A11) - A01.T*U01*U11.-1
 */
static
int blk_invldl_upper(armas_x_dense_t * A, armas_x_dense_t * W, int lb,
                     armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A01, A11, A22, U01, w2;
    int err = 0;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);

    while (ABR.rows > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            __nil, &A11, __nil,
            __nil, __nil, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        // ---------------------------------------------------------------------
        // workspaces;
        armas_x_make(&w2, lb, 1, lb, armas_x_data(W));
        armas_x_make(&U01, A01.rows, A11.cols, A01.rows, &armas_x_data(W)[lb]);
        armas_x_mcopy(&U01, &A01, 0, conf);

        // A01 = - A00*U01*U11.-1
        armas_x_mult_sym(ZERO, &A01, -ONE, &A00, &U01,
                         ARMAS_LEFT | ARMAS_UPPER, conf);
        armas_x_solve_trm(&A01, ONE, &A11,
                          ARMAS_RIGHT | ARMAS_UPPER | ARMAS_UNIT, conf);

        // A11 = inv(A11) - A01.T*U01*U11.-1
        armas_x_solve_trm(&U01, ONE, &A11,
                          ARMAS_RIGHT | ARMAS_UPPER | ARMAS_UNIT, conf);
        if (unblk_invldl_upper(&A11, &w2, conf) < 0 && err == 0)
            err = -1;
        armas_x_update_trm(ONE, &A11, -ONE, &A01, &U01,
                           ARMAS_UPPER | ARMAS_TRANSA, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
    }
    return err;
}


/**
 * @brief Inverse of symmetric LDL^T or UDY^T factorized matrix
 *
 * @param[in,out] A
 *     On entry, the LDL^T (UDU^T) factorized symmetric matrix.
 *     On exit inverse of the matrix.
 * @param[in] P
 *     Pivot vector
 * @param[in] flags
 *     Indicator flags, if ARMAS_LOWER (ARMAS_UPPER) is set then the elements
 *     of the symmetric matrix are store in the lower (upper) triangular part
 *     of the matrix. The strictly  upper (lower) part is not accessed.
 * @param[out]  W
 *     Workspace, miminum space needed is N elements.
 * @param[in,out] conf
 *     Configuration options, error status
 *
 * @retval  0   OK
 * @retval -1   failure, conf.error holds actual error code
 */
int armas_x_ldlinverse_w(armas_x_dense_t * A, const armas_pivot_t * P,
                         int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    int lb, err = 0;
    size_t wsmin, wsz;
    armas_x_dense_t Wt;
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
            wb->bytes = (env->lb + 1) * A->rows * sizeof(DTYPE);
        else
            wb->bytes = A->rows * sizeof(DTYPE);
        return 0;
    }

    if (A->rows != A->cols) {
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
        if (wsz < (lb + 1) * A->rows) {
            lb = (wsz / A->rows - 1) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    armas_x_make(&Wt, wsz, 1, wsz, (DTYPE *) armas_wptr(wb));
    wsz = armas_wpos(wb);

    if (lb == 0 || A->rows <= lb) {
        if (flags & ARMAS_UPPER)
            err = unblk_invldl_upper(A, &Wt, conf);
        else
            err = unblk_invldl_lower(A, &Wt, conf);
    } else {
        if (flags & ARMAS_UPPER)
            err = blk_invldl_upper(A, &Wt, lb, conf);
        else
            err = blk_invldl_lower(A, &Wt, lb, conf);
    }

    if (P != ARMAS_NOPIVOT) {
        if (flags & ARMAS_UPPER) {
            armas_x_pivot(A, (armas_pivot_t *) P,
                          ARMAS_PIVOT_UPPER | ARMAS_PIVOT_FORWARD, conf);
        } else {
            armas_x_pivot(A, (armas_pivot_t *) P,
                          ARMAS_PIVOT_LOWER | ARMAS_PIVOT_BACKWARD, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return err;
}


int armas_x_ldlinverse(armas_x_dense_t * A,
                       const armas_pivot_t * P, int flags, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (!conf)
        conf = armas_conf_default();

    if (armas_x_ldlinverse_w(A, P, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_x_ldlinverse_w(A, P, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;
}

#endif                          /* ARMAS_PROVIDES && ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
