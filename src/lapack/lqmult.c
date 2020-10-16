
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_lqmult) && defined(armas_lqmult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_householder)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


/*
 * Unblocked algorith for computing C = Q.T*C and C = Q*C.
 *
 * Q = H(k)H(k-1)...H(1) where elementary reflectors H(i) are stored on i'th row
 * right of diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller row numbers
 * to larger, produces H(k)...H(2)H(1) == Q. and C = Q*C
 *
 * Progressing from bottom-right to top-left produces H(k)H(k-1)...H(1) == Q.T and C = Q.T*C
 */
static
int unblk_lqmult_left(armas_dense_t * C, armas_dense_t * A,
                      armas_dense_t * tau, armas_dense_t * W, int flags,
                      armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a11, a12, A22, *Aref;
    armas_dense_t tT, tB, t0, t1, t2, w12;
    armas_dense_t CT, CB, C0, c1, C2;
    int pAdir, pAstart, pStart, pDir;
    int mb, nb, tb, cb;

    EMPTY(A00);
    EMPTY(a11);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->rows - A->rows);
        tb = max(0, armas_size(tau) - A->rows);
        Aref = &ATL;
    } else {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = nb = tb = cb = 0;
        Aref = &ABR;
    }

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_2x1(
        &CT, &CB, /**/ C, cb, pStart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    armas_make(&w12, C->cols, 1, C->cols, armas_data(W));

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, pAdir);
        mat_repartition_2x1to3x1(
            &CT, &C0, &c1, &C2, /**/ C, 1, pDir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, pDir);
        // ---------------------------------------------------------------------
        armas_apply_householder2x1(&t1, &a12,
                                     &c1, &C2, &w12, ARMAS_LEFT, conf);
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
int blk_lqmult_left(armas_dense_t * C, armas_dense_t * A,
                    armas_dense_t * tau, armas_dense_t * T,
                    armas_dense_t * W, int flags, int lb, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A11, A12, A22, AR, *Aref;
    armas_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_dense_t CT, CB, C0, C1, C2;
    int pAdir, pAstart, pStart, pDir;
    int mb, nb, tb, cb, transpose;

    EMPTY(A00);
    EMPTY(C0);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->rows - A->rows);
        tb = max(0, armas_size(tau) - min(A->rows, A->cols));
        Aref = &ATL;
        transpose = FALSE;
    } else {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = nb = tb = cb = 0;
        Aref = &ABR;
        transpose = TRUE;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_2x1(
        &CT, &CB, /**/ C, cb, pStart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, __nil, &A22, /**/ A, lb, pAdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, lb, pDir);
        mat_repartition_2x1to3x1(
            &CT, &C0, &C1, &C2, /**/ C, A11.cols, pDir);
        // ---------------------------------------------------------------------
        // build block reflector
        mat_merge1x2(&AR, &A11, &A12);
        armas_make(&Tcur, A11.cols, A11.cols, A11.cols, armas_data(T));
        armas_mscale(&Tcur, ZERO, 0, conf);
        armas_unblk_lq_reflector(&Tcur, &AR, &t1, conf);

        // compute Q*C or Q.T*C
        armas_make(&Wrk, C1.cols, A11.cols, C1.cols, armas_data(W));
        armas_update_lq_left(&C1, &C2,
                               &A11, &A12, &Tcur, &Wrk, transpose, conf);
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
 * Q = H(k)H(k-1)...H(1) where elementary reflectors H(i) are stored on i'th row
 * right of diagonal in A.
 *
 *     Q.T = (H1(k)*...H(2)*H(1)).T
 *         = H(1).T*H(2)*T...*H(1).T
 *         = H(1)H(2)...H(k)
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces C*H(1)H(2)...H(k) == C*Q.T.
 *
 * Progressing from bottom-right to top-left produces C*H(k)...H(2)H(1) == C*Q.
 */
static
int unblk_lqmult_right(armas_dense_t * C, armas_dense_t * A,
                       armas_dense_t * tau, armas_dense_t * W, int flags,
                       armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a11, a12, A22, *Aref;
    armas_dense_t tT, tB, t0, t1, t2, w12;
    armas_dense_t CL, CR, C0, c1, C2;
    int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
    int mb, nb, tb, cb;

    EMPTY(C0);
    EMPTY(CL);
    EMPTY(A00);
    EMPTY(a11);

    if (flags & ARMAS_TRANS) {
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pCstart = ARMAS_PLEFT;
        pCdir = ARMAS_PRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        mb = nb = tb = cb = 0;
        Aref = &ABR;
    } else {
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pCstart = ARMAS_PRIGHT;
        pCdir = ARMAS_PLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->cols - A->rows);
        tb = max(0, armas_size(tau) - min(A->rows, A->cols));
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
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, pAdir);
        mat_repartition_1x2to1x3(
            &CL, &C0, &c1, &C2, /**/ C, 1, pCdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, pDir);
        // ---------------------------------------------------------------------
        armas_apply_householder2x1(&t1, &a12,
                                     &c1, &C2, &w12, ARMAS_RIGHT, conf);
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
 * Blocked version for computing C = C*Q and C = C*Q.T from elementary
 *  reflectors and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block
 * reflector T. Matrix C is updated by applying block reflector T using
 * compact WY algorithm.
 */
static
int blk_lqmult_right(armas_dense_t * C, armas_dense_t * A,
                     armas_dense_t * tau, armas_dense_t * T,
                     armas_dense_t * W, int flags, int lb,
                     armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A11, A12, A21, A22, AR, *Aref;
    armas_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_dense_t CL, CR, C0, C1, C2;
    int pAdir, pAstart, pStart, pDir, pCstart, pCdir;
    int mb, nb, cb, tb, transpose;

    EMPTY(A00);
    EMPTY(C0);
    EMPTY(CL);

    if (flags & ARMAS_TRANS) {
        // from top-left to bottom-right to produce transpose sequence (C*Q.T)
        pAstart = ARMAS_PTOPLEFT;
        pAdir = ARMAS_PBOTTOMRIGHT;
        pStart = ARMAS_PTOP;
        pDir = ARMAS_PBOTTOM;
        pCstart = ARMAS_PLEFT;
        pCdir = ARMAS_PRIGHT;
        mb = cb = tb = nb = 0;
        Aref = &ABR;
        transpose = TRUE;
    } else {
        // from bottom-right to top-left to produce normal sequence (C*Q)
        pAstart = ARMAS_PBOTTOMRIGHT;
        pAdir = ARMAS_PTOPLEFT;
        pStart = ARMAS_PBOTTOM;
        pDir = ARMAS_PTOP;
        pCstart = ARMAS_PRIGHT;
        pCdir = ARMAS_PLEFT;
        mb = max(0, A->rows - A->cols);
        nb = max(0, A->cols - A->rows);
        cb = max(0, C->cols - A->rows);
        tb = max(0, armas_size(tau) - min(A->rows, A->cols));
        Aref = &ATL;
        transpose = FALSE;
    }

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_1x2(
        &CL, &CR, /**/ C, cb, pCstart);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, tb, pStart);

    while (Aref->rows > 0 && Aref->cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, &A21, &A22, /**/ A, lb, pAdir);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, lb, pDir);
        mat_repartition_1x2to1x3(
            &CL, &C0, &C1, &C2, /**/ C, A11.cols, pCdir);
        // ---------------------------------------------------------------------
        // build block reflector
        mat_merge1x2(&AR, &A11, &A12);
        armas_make(&Tcur, A11.cols, A11.cols, A11.cols, armas_data(T));
        armas_mscale(&Tcur, ZERO, 0, conf);
        armas_unblk_lq_reflector(&Tcur, &AR, &t1, conf);

        // compute Q*C or Q.T*C
        armas_make(&Wrk, C1.rows, A11.cols, C1.rows, armas_data(W));
        armas_update_lq_right(&C1, &C2,
                                &A11, &A12, &Tcur, &Wrk, transpose, conf);
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
 * @brief Multiply with orthogonal matrix Q from LQ factorization
 *
 * @see armas_lqmult_w
 * @ingroup lapack
 */
int armas_lqmult(armas_dense_t * C,
                   const armas_dense_t * A,
                   const armas_dense_t * tau, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_lqmult_w(C, A, tau, flags, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -ARMAS_EMEMORY;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_lqmult_w(C, A, tau, flags, wbs, cf);
    armas_wrelease(&wb);
    return err;
}


/**
 * @brief Multiply with orthogonal matrix Q from LQ factorization
 *
 * Multiply and replace C with \f$ Q*C \f$ or \f$ Q^T*C  \f$ where Q is a
 * real orthogonal matrix defined as the product of k elementary reflectors.
 *
 *    \f$ Q = H(k)H(k-1)...H(1) \f$
 *
 * as returned by armas_lqfactor_w().
 *
 * @param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit RIGHT is set then
 *     N-by-M matrix.  On exit C is overwritten by \f$ Q*C or Q^T*C \f$.
 *     If bit RIGHT is set then C is  overwritten by \f$ C*Q or C*Q^T. \f$
 *
 * @param[in] A
 *     LQ factorization as returned by armas_lqfactor_w() where the upper
 *     trapezoidal part holds the elementary reflectors.
 *
 * @param[in] tau
 *   The scalar factors of the elementary reflectors.
 *
 * @param[in] flags
 *     Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT*, *ARMAS_TRANS*
 *
 * @param[out] wb
 *    Workspace buffer needed for computation. If *wb.bytes* is zero then the
 *    required workspace size is computed and returned immediately.
 *
 * @param[in,out] conf
 *     Blocking configuration.
 *
 * @retval  0 Success
 * @retval <0 Error, `conf.error` holds error code
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if m(C) != n(A) for C*op(Q) or n(C) != n(A) for op(Q)*C
 *   - `ARMAS_EINVAL` C or A or tau is null pointer
 *   - `ARMAS_EWORK`  if workspace is less than required for unblocked computation
 *
 * Compatible with lapack.DORMLQ
 *
 * Notes
 *   m(A) is number of elementary reflectors == A.rows
 *   n(A) is the order of the Q matrix == A.cols
 *
 * @cond
 *   LEFT : m(C) == n(A)
 *   RIGHT: n(C) == n(A)
 * @endcond
 * @ingroup lapack
 */
int armas_lqmult_w(armas_dense_t * C,
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
        return -1;
    }
    env = armas_getenv();
    K = (flags & ARMAS_RIGHT) != 0 ? C->rows : C->cols;
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && K > env->lb)
            wb->bytes = ((env->lb + K) * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = K * sizeof(DTYPE);
        return 0;
    }

    if (!A || !tau) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    // check sizes; A, tau return from armas_qrfactor()
    P = (flags & ARMAS_RIGHT) != 0 ? C->cols : C->rows;
    if (P != A->cols || armas_size(tau) != A->rows) {
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
        if (wsz < (K + lb) * lb) {
            // ws =   (K + lb)*lb => lb^2 + K*lb - wsz = 0
            //    =>  (sqrt(K^2 + 4*wsz) - K)/2
            lb = ((int) (SQRT((DTYPE) (K * K + 4 * wsz))) - K) / 2;
            lb &= ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || K <= lb) {
        // unblocked 
        armas_make(&Wrk, K, 1, K, buf);
        if ((flags & ARMAS_RIGHT) != 0) {
            unblk_lqmult_right(C, (armas_dense_t *) A,
                                 (armas_dense_t *) tau, &Wrk, flags, conf);
        } else {
            unblk_lqmult_left(C, (armas_dense_t *) A,
                                (armas_dense_t *) tau, &Wrk, flags, conf);
        }
    } else {
        // blocked code; block reflector T and temporary space
        armas_make(&T, lb, lb, lb, buf);
        armas_make(&Wrk, K, lb, K, &buf[armas_size(&T)]);

        if ((flags & ARMAS_RIGHT) != 0) {
            blk_lqmult_right(C, (armas_dense_t *) A,
                               (armas_dense_t *) tau, &T, &Wrk, flags, lb,
                               conf);
        } else {
            blk_lqmult_left(C, (armas_dense_t *) A, (armas_dense_t *) tau,
                              &T, &Wrk, flags, lb, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
