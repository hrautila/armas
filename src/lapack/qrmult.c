
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! QR orthogonal matrix multiplication

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_qrmult) && defined(armas_x_qrmult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas1)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"
//! \endcond


#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


/*
 * Unblocked algorith for computing C = Q.T*C and C = Q*C.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th
 * column below diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces H(k)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(k) == Q
 * and C = Q*C
 */
static
int unblk_qrmult_left(armas_x_dense_t * C, armas_x_dense_t * A,
                      armas_x_dense_t * tau, armas_x_dense_t * W, int flags,
                      armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, w12;
    armas_x_dense_t CT, CB, C0, c1, C2;
    int pAdir, pAstart, pStart, pDir;
    int mb, nb, tb;

    EMPTY(A00);
    EMPTY(a11);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = nb = tb = 0;
        Aref = &ABR;
    } else {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        tb = max(0, armas_x_size(tau) - A->cols);
        Aref = &ATL;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_2x1(
        &CT,
        &CB, /**/ C, mb, pStart);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tau, tb, pStart);

    armas_x_make(&w12, C->cols, 1, C->cols, armas_x_data(W));

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, pAdir);
        mat_repartition_2x1to3x1(
            &CT, &C0, &c1, &C2, /**/ C, 1, pDir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, pDir);
        // --------------------------------------------------------------------
        armas_x_apply_householder2x1(&t1, &a21, &c1, &C2, &w12,
                                     ARMAS_LEFT, conf);
        // --------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, pAdir);
        mat_continue_3x1to2x1(
            &CT, &CB, /**/ &C0, &c1, C, pDir);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, pDir);
    }
    return 0;
}

static
int blk_qrmult_left(armas_x_dense_t * C, armas_x_dense_t * A,
                    armas_x_dense_t * tau, armas_x_dense_t * T,
                    armas_x_dense_t * W, int flags, int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A10, A11, A20, A21, A22, AL, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_x_dense_t CT, CB, C0, C1, C2;
    int pAdir, pAstart, pStart, pDir;
    int mb, nb, tb, transpose;

    // initialize to GCC "maybe-uninitialized" error
    EMPTY(A00);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = nb = tb = 0;
        Aref = &ABR;
        transpose = TRUE;
    } else {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        tb = max(0, armas_x_size(tau) - IMIN(A->rows, A->cols));
        Aref = &ATL;
        transpose = FALSE;
    }

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_2x1(
        &CT,
        &CB, /**/ C, mb, pStart);
    mat_partition_2x1(
        &tT,
        &tB, /**/ tau, tb, pStart);

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &A10, &A11, __nil,
            &A20, &A21, &A22, /**/ A, lb, pAdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, A11.cols, pDir);
        mat_repartition_2x1to3x1(
            &CT, &C0, &C1, &C2, /**/ C, A11.cols, pDir);
        // ---------------------------------------------------------------------
        // build block reflector
        mat_merge2x1(&AL, &A11, &A21);
        armas_x_make(&Tcur, A11.cols, A11.cols, A11.cols, armas_x_data(T));
        armas_x_mscale(&Tcur, ZERO, 0, conf);
        armas_x_unblk_qr_reflector(&Tcur, &AL, &t1, conf);

        // compute Q*C or Q.T*C
        armas_x_make(&Wrk, C1.cols, A11.cols, C1.cols, armas_x_data(W));
        armas_x_update_qr_left(&C1, &C2, &A11, &A21, &Tcur,
                               &Wrk, transpose, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, pAdir);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, pDir);
        mat_continue_3x1to2x1(
            &CT, &CB, /**/ &C0, &C1, C, pDir);
    }
    return 0;
}

/*
 * Unblocked algorith for computing C = C*Q.T and C = C*Q.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th column
 * below diagonal in A.
 *
 *     Q.T = (H1(1)*H(2)*...*H(k)).T
 *         = H(k).T*...*H(2).T*H(1).T
 *         = H(k)...H(2)H(1)
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces C*H(1)H(2)...H(k) == C*Q.
 *
 * Progressing from bottom-right to top-left produces C*H(k)...H(2)H(1) == C*Q.T.
 */
static
int unblk_qrmult_right(armas_x_dense_t * C, armas_x_dense_t * A,
                       armas_x_dense_t * tau, armas_x_dense_t * W, int flags,
                       armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, a21, A22, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, w12;
    armas_x_dense_t CL, CR, C0, c1, C2;
    int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
    int mb, nb, tb, cb;

    EMPTY(A00);
    EMPTY(a11);
    EMPTY(C0);
    EMPTY(CL);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pCstart = ARMAS_PRIGHT;
        pCdir = ARMAS_PLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->cols - A->cols);
        tb = max(0, armas_x_size(tau) - IMIN(A->rows, A->cols));
        Aref = &ATL;
    } else {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pCstart = ARMAS_PLEFT;
        pCdir = ARMAS_PRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = nb = tb = cb = 0;
        Aref = &ABR;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_1x2(
        &CL, &CR, /**/ C, cb, pCstart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    armas_x_make(&w12, C->rows, 1, C->rows, armas_x_data(W));

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, pAdir);
        mat_repartition_1x2to1x3(
            &CL, &C0, &c1, &C2, /**/ C, 1, pCdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, pDir);
        // ---------------------------------------------------------------------
        armas_x_apply_householder2x1(&t1, &a21, &c1, &C2, &w12,
                                     ARMAS_RIGHT, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, pAdir);
        mat_continue_1x3to1x2(
            &CL, &CR, /**/ &C0, &c1, C, pCdir);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, pDir);
    }
    return 0;
}

/*
 * Blocked version for computing C = C*Q and C = C*Q.T from elementary reflectors
 * and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block reflector T.
 * Matrix C is updated by applying block reflector T using compact WY algorithm.
 */
static
int blk_qrmult_right(armas_x_dense_t * C, armas_x_dense_t * A,
                     armas_x_dense_t * tau, armas_x_dense_t * T,
                     armas_x_dense_t * W, int flags,
                     int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A10, A11, A20, A21, A22, AL, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_x_dense_t CL, CR, C0, C1, C2;
    int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
    int mb, nb, cb, tb, transpose;
    DTYPE *wdata = armas_x_data(W);

    // initialize to "empty" to avoid "maybe-uninitialized" errors
    EMPTY(A00);
    EMPTY(A11);
    EMPTY(A22);
    EMPTY(CL);
    EMPTY(CR);
    EMPTY(C0);

    if (flags & ARMAS_TRANS) {
        // from bottom-right to top-left to produce transpose sequence (C*Q.T)
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        pCstart = ARMAS_PRIGHT;
        pCdir = ARMAS_PLEFT;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->cols - A->cols);
        tb = max(0, armas_x_size(tau) - IMIN(A->rows, A->cols));
        Aref = &ATL;
        transpose = TRUE;
    } else {
        // from top-left to bottom-right to produce normal sequence (C*Q)
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        pCstart = ARMAS_PLEFT;
        pCdir = ARMAS_PRIGHT;
        mb = cb = tb = nb = 0;
        Aref = &ABR;
        transpose = FALSE;
    }

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_1x2(
        &CL, &CR, /**/ C, cb, pCstart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &A10, &A11, __nil,
            &A20, &A21, &A22, /**/ A, lb, pAdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, A11.cols, pDir);
        mat_repartition_1x2to1x3(
            &CL, &C0, &C1, &C2, /**/ C, A11.cols, pCdir);
        // ---------------------------------------------------------------------
        // build block reflector
        mat_merge2x1(&AL, &A11, &A21);
        armas_x_make(&Tcur, A11.cols, A11.cols, A11.cols, armas_x_data(T));
        armas_x_mscale(&Tcur, 0.0, 0, conf);
        armas_x_unblk_qr_reflector(&Tcur, &AL, &t1, conf);

        // compute Q*C or Q.T*C
        armas_x_make(&Wrk, C1.rows, A11.cols, C1.rows, wdata);
        armas_x_update_qr_right(&C1, &C2, &A11, &A21, &Tcur,
                                &Wrk, transpose, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, pAdir);
        mat_continue_1x3to1x2(
            &CL, &CR, /**/ &C0, &C1, C, pCdir);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, pDir);
    }
    return 0;
}

/**
 * @brief Multiply matrix with orthogonal matrix Q.
 *
 * Multiply and replace C with \f$ Q C \f$ or \f$ Q^T C \f$ where Q is a real
 *  orthogonal matrix defined as the product of k elementary reflectors.
 *
 *    \f$ Q = H_1 H_2 . . . H_k \f$
 *
 * as returned by qrfactor().
 *
 * @param[in,out] C
 *   On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then
 *   N-by-M matrix On exit C is overwritten by \f$ Q C \f$ or \f$ Q^T C \f$.
 *   If bit *ARMAS_RIGHT* is  set then C is overwritten by \f$ C Q \f$
 *   or \f$ CQ^T \f$.
 *
 * @param[in] A
 *    QR factorization as returne by qrfactor() where the lower trapezoidal
 *    part holds the elementary reflectors.
 *
 * @param[in] tau
 *   The scalar factors of the elementary reflectors.
 *
 * @param[in,out] W
 *    Workspace matrix, required size is returned by qrmult_work().
 *
 * @param[in] flags
 *   Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT*, *ARMAS_TRANS*
 *
 * @param conf
 *   Blocking configuration. Field LB defines block size. If it is zero
 *   unblocked invocation is assumed. Actual blocking size is adjusted
 *   to available workspace size and minimum of configured block size and
 *   block size implied by workspace is used.
 *
 * Compatible with lapack.DORMQR
 * \ingroup lapack
 */
int armas_x_qrmult(armas_x_dense_t * C,
                   const armas_x_dense_t * A,
                   const armas_x_dense_t * tau, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_x_qrmult_w(C, A, tau, flags, &wb, cf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;
    int stat = armas_x_qrmult_w(C, A, tau, flags, wbs, cf);
    armas_wrelease(&wb);
    return stat;
}


/**
 * @brief Multiply matrix with orthogonal matrix Q.
 *
 * Multiply and replace C with \f$ Q C \f$ or \f$ Q^T C \f$ where Q is a real
 * orthogonal matrix defined as the product of k elementary reflectors.
 *
 *    \f$ Q = H_1 H_2 . . . H_k \f$
 *
 * as returned by qrfactor().
 *
 * @param[in,out] C
 *   On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then N-by-M matrix
 *   On exit C is overwritten by \f$ Q C \f$ or \f$ Q^T C \f$. If bit *ARMAS_RIGHT* is
 *   set then C is overwritten by \f$ C Q \f$ or \f$ CQ^T \f$.
 *
 * @param[in] A
 *    QR factorization as returne by qrfactor() where the lower trapezoidal
 *    part holds the elementary reflectors.
 *
 * @param[in] tau
 *   The scalar factors of the elementary reflectors.
 *
 * @param[in] flags
 *   Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT*, *ARMAS_TRANS*
 *
 * @param[in,out] wb
 *    Workspace buffer needed for computation. To compute size of the required
 *    space call the function with workspace bytes set to zero. Size of
 *    workspace is returned in `wb.bytes` and no other computation or parameter
 *    size checking is done and function returns with success.
 *
 * @param conf
 *   Blocking configuration. Field LB defines blocking size. If it is zero
 *   unblocked invocation is assumed. Actual blocking size is adjusted
 *   to available workspace size and minimum of configured block size and
 *   block size implied by workspace is used.
 *
 *  @retval 0  success
 *  @retval -1 error and `conf.error` set to last error
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if n(C) != m(A) for C*op(Q) or m(C) != m(A) for op(Q)*C
 *   - `ARMAS_EINVAL` C or A or tau is null pointer
 *   - `ARMAS_EWORK`  if workspace is less than required for unblocked computation
 *
 * Compatible with lapack.xORMQR
 * \ingroup lapack
 */
int armas_x_qrmult_w(armas_x_dense_t * C,
                     const armas_x_dense_t * A,
                     const armas_x_dense_t * tau,
                     int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t T, Wrk;
    size_t wsmin, wsz = 0;
    int lb, K, P;
    DTYPE *buf;
    armas_x_dense_t tauh;
    armas_env_t *env;

    if (!conf)
        conf = armas_conf_default();

    if (!C) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    env = armas_getenv();
    K = (flags & ARMAS_RIGHT) != 0 ? C->rows : C->cols;
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && K > env->lb)
            wb->bytes = ((K + env->lb) * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = K * sizeof(DTYPE);
        return 0;
    }

    if (!A || !tau) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    // check sizes; A, tau return from armas_x_qrfactor()
    P = (flags & ARMAS_RIGHT) != 0 ? C->cols : C->rows;
    if (P != A->rows || armas_x_size(tau) != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    lb = env->lb;
    wsmin = K * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // adjust blocking factor for workspace
    if (lb > 0 && K > lb) {
        wsz /= sizeof(DTYPE);
        // ws = (K + lb)*lb => lb^2 + K*lb - wsz = 0  =>  (sqrt(K^2 + 4*wsz) - K)/2
        if (wsz < (K + lb) * lb) {
            lb = ((int) (SQRT((DTYPE) (K * K + 4 * wsz))) - K) / 2;
            lb &= ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    EMPTY(tauh);
    armas_x_submatrix(&tauh, tau, 0, 0, A->cols, 1);
    if (lb == 0 || K <= lb) {
        // unblocked 
        armas_x_make(&Wrk, K, 1, K, buf);
        if (flags & ARMAS_RIGHT) {
            unblk_qrmult_right(C, (armas_x_dense_t *) A, &tauh, &Wrk, flags,
                                 conf);
        } else {
            unblk_qrmult_left(C, (armas_x_dense_t *) A, &tauh, &Wrk, flags,
                                conf);
        }
    } else {
        // blocked code; block reflector T and temporary space
        armas_x_make(&T, lb, lb, lb, buf);
        armas_x_make(&Wrk, K, lb, K, &buf[armas_x_size(&T)]);

        if (flags & ARMAS_RIGHT) {
            blk_qrmult_right(C, (armas_x_dense_t *) A, &tauh, &T, &Wrk, flags,
                               lb, conf);
        } else {
            blk_qrmult_left(C, (armas_x_dense_t *) A, &tauh, &T, &Wrk, flags,
                              lb, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
