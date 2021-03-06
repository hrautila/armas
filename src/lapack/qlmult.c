
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Multiply with Q matrix

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_qlmult)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_householder)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


static
int unblk_qlmult_left(armas_dense_t * C, armas_dense_t * A,
                      armas_dense_t * tau, armas_dense_t * W, int flags,
                      armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a01, a11, A22, *Aref;
    armas_dense_t tT, tB, t0, t1, t2, w12;
    armas_dense_t CT, CB, C0, c1, C2;
    int pAdir, pAstart, pStart, pDir;
    int mb, nb, tb;

    EMPTY(A00);
    EMPTY(a11);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = nb = tb = 0;
        Aref = &ATL;
    } else {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        tb = max(0, armas_size(tau) - A->cols);
        Aref = &ABR;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_2x1(
        &CT, &CB, /**/ C, mb, pStart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    armas_make(&w12, C->cols, 1, C->cols, armas_data(W));

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, pAdir);
        mat_repartition_2x1to3x1(
            &CT, &C0, &c1, &C2, /**/ C, 1, pDir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, pDir);
        // ---------------------------------------------------------------------
        armas_apply_householder2x1(&t1, &a01,
                                     &c1, &C0, &w12, ARMAS_LEFT, conf);
        // ---------------------------------------------------------------------
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
int blk_qlmult_left(armas_dense_t * C, armas_dense_t * A,
                    armas_dense_t * tau, armas_dense_t * T,
                    armas_dense_t * W, int flags, int lb, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A01, A11, A22, *Aref, AT;
    armas_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_dense_t CT, CB, C0, C1, C2;
    int pAdir, pAstart, pStart, pDir;
    int mb, nb, tb, transpose;

    EMPTY(A00);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = nb = tb = 0;
        Aref = &ATL;
        transpose = TRUE;
    } else {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        tb = max(0, armas_size(tau) - A->cols);
        Aref = &ABR;
        transpose = FALSE;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_2x1(
        &CT, &CB, /**/ C, mb, pStart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            __nil, &A11, __nil,
            __nil, __nil, &A22, /**/ A, lb, pAdir);
        mat_repartition_2x1to3x1(
            &CT, &C0, &C1, &C2, /**/ C, A11.cols, pDir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, A11.cols, pDir);
        // ---------------------------------------------------------------------
        // block reflector for current block
        mat_merge2x1(&AT, &A01, &A11);
        armas_make(&Tcur, A11.cols, A11.cols, A11.cols, armas_data(T));
        armas_mscale(&Tcur, ZERO, 0, conf);
        armas_unblk_ql_reflector(&Tcur, &AT, &t1, conf);

        // update with (I - Y*T*Y.T) or (I - Y*T*Y.T).T
        armas_make(&Wrk, C1.cols, A11.cols, C1.cols, armas_data(W));
        armas_update_ql_left(&C1, &C0,
                               &A11, &A01, &Tcur, &Wrk, transpose, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, pAdir);
        mat_continue_3x1to2x1(
            &CT, &CB, /**/ &C0, &C1, C, pDir);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, pDir);
    }
    return 0;
}

/*
 * Compute C := C*Q or C := C*Q.T where Q is the M-by-N orthogonal matrix
 * defined K elementary reflectors. (K = A.cols)
 */
static
int unblk_qlmult_right(armas_dense_t * C, armas_dense_t * A,
                       armas_dense_t * tau, armas_dense_t * W, int flags,
                       armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a01, a11, A22, *Aref;
    armas_dense_t tT, tB, t0, t1, t2, w12;
    armas_dense_t CL, CR, C0, c1, C2;
    int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
    int mb, nb, tb, cb;

    EMPTY(A00);
    EMPTY(a11);
    EMPTY(CL);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pCstart = ARMAS_PLEFT;
        pCdir = ARMAS_PRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->cols - A->cols);
        tb = max(0, armas_size(tau) - min(A->rows, A->cols));
        Aref = &ABR;
    } else {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pCstart = ARMAS_PRIGHT;
        pCdir = ARMAS_PLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = nb = tb = cb = 0;
        Aref = &ATL;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_1x2(
        &CL, &CR, /**/ C, cb, pCstart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    armas_make(&w12, C->rows, 1, C->rows, armas_data(W));

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            __nil, &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, pAdir);
        mat_repartition_1x2to1x3(
            &CL, &C0, &c1, &C2, /**/ C, 1, pCdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, pDir);
        // ---------------------------------------------------------------------
        armas_apply_householder2x1(&t1, &a01,
                                     &c1, &C0, &w12, ARMAS_RIGHT, conf);
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

static
int blk_qlmult_right(armas_dense_t * C, armas_dense_t * A,
                     armas_dense_t * tau, armas_dense_t * T,
                     armas_dense_t * W, int flags, int lb,
                     armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A01, A11, A22, *Aref, AT;
    armas_dense_t tT, tB, t0, t1, t2, w12, Tcur, Wrk;
    armas_dense_t CL, CR, C0, C1, C2;
    int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
    int mb, nb, tb, cb, transpose;

    EMPTY(A00);
    EMPTY(CL);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pCstart = ARMAS_PLEFT;
        pCdir = ARMAS_PRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->cols - A->cols);
        tb = max(0, armas_size(tau) - min(A->rows, A->cols));
        Aref = &ABR;
        transpose = TRUE;
    } else {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pCstart = ARMAS_PRIGHT;
        pCdir = ARMAS_PLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = nb = tb = cb = 0;
        Aref = &ATL;
        transpose = FALSE;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_1x2(
        &CL, &CR, /**/ C, mb, pCstart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    armas_submatrix(&w12, W, 0, 0, C->rows, 1);

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            __nil, &A11, __nil,
            __nil, __nil, &A22, /**/ A, lb, pAdir);
        mat_repartition_1x2to1x3(
            &CL, &C0, &C1, &C2, /**/ C, A11.cols, pCdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, lb, pDir);
        // ---------------------------------------------------------------------
        // build reflector
        mat_merge2x1(&AT, &A01, &A11);
        armas_make(&Tcur, A11.cols, A11.cols, A11.cols, armas_data(T));
        armas_mscale(&Tcur, ZERO, 0, conf);
        armas_unblk_ql_reflector(&Tcur, &AT, &t1, conf);

        // update with current block
        armas_make(&Wrk, C1.rows, A11.cols, C1.rows, armas_data(W));
        armas_update_ql_right(&C1, &C0,
                                &A11, &A01, &Tcur, &Wrk, transpose, conf);
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
 * @brief Multiply with orthogonal Q matrix
 *
 * @see armas_qlmult_w
 * @ingroup lapack
 */
int armas_qlmult(armas_dense_t * C,
                   const armas_dense_t * A,
                   const armas_dense_t * tau, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_qlmult_w(C, A, tau, flags, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -ARMAS_EMEMORY;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_qlmult_w(C, A, tau, flags, wbs, cf);
    armas_wrelease(&wb);
    return err;
}


/**
 * @brief Multiply with orthogonal Q matrix 
 *
 * Multiply and replace C with \f$ QC \f$ or \f$ Q^TC \f$ where Q is a real
 * orthogonal matrix defined as the product of K first elementary reflectors.
 *
 *    \f$ Q = H_k H_{k-1} ... H_k \f$
 *
 * as returned by armas_qlfactor().
 *
 * @param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit *ARMAS_RIGHT* is set then
 *     N-by-M matrix On exit C is overwritten by \f$ QC \f$ or \f$ Q^TC \f$. If
 *     bit *ARMAS_LEFT* is set then C is overwritten by \f$ CQ \f$ or
 *     \f$ CQ^T \f$
 *
 * @param[in] A
 *     An N-by-K QL factorization as returned by armas_qlfactor() where the
 *     upper trapezoidal part holds the elementary reflectors.
 *
 * @param[in] tau
 *    The scalar factors of the elementary reflectors, vector of length n(A)
 *
 * @param[in] flags
 *    Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT* and *ARMAS_TRANS*
 *
 * @param wb
 *    Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *    immediately.
 *
 * @param[in] conf
 *   Configuration options.
 *
 * @retval  0 Succes
 * @retval <0 Failure, `conf.error` holds error code.
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if n(C) != m(A) for C*op(Q) or m(C) != m(A) for op(Q)*C
 *   - `ARMAS_EINVAL` C or A or tau is null pointer
 *   - `ARMAS_EWORK`  if workspace is less than required for unblocked computation
 *
 * Compatible with lapack.DORMQL
 * @ingroup lapack
 */
int armas_qlmult_w(armas_dense_t * C,
                     const armas_dense_t * A,
                     const armas_dense_t * tau,
                     int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t T, Wrk;
    armas_env_t *env;
    size_t wsmin, wsz = 0;
    int lb, K, P;
    DTYPE *buf;

    if (!conf)
        conf = armas_conf_default();

    if (!C) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    env = armas_getenv();
    K = (flags & ARMAS_RIGHT) != 0 ? C->rows : C->cols;
    if (wb && wb->bytes == 0) {
        if (env->lb > 0)
            wb->bytes = ((K + env->lb) * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = K * sizeof(DTYPE);
        return 0;
    }
    if (!A || !tau) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    // check sizes; A, tau return from armas_qrfactor()
    P = (flags & ARMAS_RIGHT) != 0 ? C->cols : C->rows;
    if (P != A->rows || armas_size(tau) != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }


    lb = env->lb;
    wsmin = K * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    // adjust blocking factor for workspace
    if (lb > 0 && K > lb) {
        wsz /= sizeof(DTYPE);
        if (wsz < (K + lb) * lb) {
            // ws = (K + lb)*lb => lb^2 + K*lb - wsz = 0  =>  (sqrt(K^2 + 4*wsz) - K)/2
            lb = ((int) (SQRT((DTYPE) (K * K + 4 * wsz))) - K) / 2;
            lb &= ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || K <= lb) {
        armas_make(&Wrk, K, 1, K, buf);
        // unblocked 
        if (flags & ARMAS_RIGHT) {
            unblk_qlmult_right(C, (armas_dense_t *) A,
                               (armas_dense_t *) tau, &Wrk, flags, conf);
        } else {
            unblk_qlmult_left(C, (armas_dense_t *) A,
                              (armas_dense_t *) tau, &Wrk, flags, conf);
        }
    } else {
        // space for block reflector;  temporary space 
        armas_make(&T, lb, lb, lb, buf);
        armas_make(&Wrk, K, lb, K, &buf[armas_size(&T)]);

        if (flags & ARMAS_RIGHT) {
            blk_qlmult_right(C, (armas_dense_t *) A,
                             (armas_dense_t *) tau, &T, &Wrk, flags, lb,
                             conf);
        } else {
            blk_qlmult_left(C, (armas_dense_t *) A, (armas_dense_t *) tau,
                            &T, &Wrk, flags, lb, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
