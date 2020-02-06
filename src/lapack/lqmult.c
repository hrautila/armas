
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_lqmult) && defined(armas_x_lqmult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_householder)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
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
int unblk_lqmult_left(armas_x_dense_t * C, armas_x_dense_t * A,
                      armas_x_dense_t * tau, armas_x_dense_t * W, int flags,
                      armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, A22, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, w12;
    armas_x_dense_t CT, CB, C0, c1, C2;
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
        tb = max(0, armas_x_size(tau) - A->rows);
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

    armas_x_submatrix(&w12, W, 0, 0, C->cols, 1);

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
        armas_x_apply_householder2x1(&t1, &a12,
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
int blk_lqmult_left(armas_x_dense_t * C, armas_x_dense_t * A,
                    armas_x_dense_t * tau, armas_x_dense_t * T,
                    armas_x_dense_t * W, int flags, int lb, armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A22, AR, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_x_dense_t CT, CB, C0, C1, C2;
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
        tb = max(0, armas_x_size(tau) - min(A->rows, A->cols));
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
        &ATL, __nil, __nil, &ABR, /**/ A, mb, nb, pAstart);
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
        armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
        armas_x_mscale(&Tcur, ZERO, 0, conf);
        armas_x_unblk_lq_reflector(&Tcur, &AR, &t1, conf);

        // compute Q*C or Q.T*C
        armas_x_submatrix(&Wrk, W, 0, 0, C1.cols, A11.cols);
        armas_x_update_lq_left(&C1, &C2,
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
int unblk_lqmult_right(armas_x_dense_t * C, armas_x_dense_t * A,
                       armas_x_dense_t * tau, armas_x_dense_t * W, int flags,
                       armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, a11, a12, A22, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, w12;
    armas_x_dense_t CL, CR, C0, c1, C2;
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
        tb = max(0, armas_x_size(tau) - min(A->rows, A->cols));
        Aref = &ATL;
    }

    mat_partition_2x2(&ATL, __nil, __nil, &ABR, /**/ A, mb, nb, pAstart);
    mat_partition_1x2(&CL, &CR, /**/ C, cb, pCstart);
    mat_partition_2x1(&tT, &tB, /**/ tau, tb, pStart);

    armas_x_submatrix(&w12, W, 0, 0, C->rows, 1);

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
        armas_x_apply_householder2x1(&t1, &a12,
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
int blk_lqmult_right(armas_x_dense_t * C, armas_x_dense_t * A,
                     armas_x_dense_t * tau, armas_x_dense_t * T,
                     armas_x_dense_t * W, int flags, int lb,
                     armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABR, A00, A11, A12, A21, A22, AR, *Aref;
    armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    armas_x_dense_t CL, CR, C0, C1, C2;
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
        tb = max(0, armas_x_size(tau) - min(A->rows, A->cols));
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
        armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
        armas_x_mscale(&Tcur, ZERO, 0, conf);
        armas_x_unblk_lq_reflector(&Tcur, &AR, &t1, conf);

        // compute Q*C or Q.T*C
        armas_x_submatrix(&Wrk, W, 0, 0, C1.rows, A11.cols);
        armas_x_update_lq_right(&C1, &C2,
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
 * \brief Multiply with orthogonal matrix Q from LQ factorization
 *
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(k)H(k-1)...H(1)
 *
 * as returned by lqfactor().
 *
 * \param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit RIGHT is set then
 *     N-by-M matrix.  On exit C is overwritten by Q*C or Q.T*C.
 *     If bit RIGHT is set then C is  overwritten by C*Q or C*Q.T
 *
 * \param[in] A
 *     LQ factorization as returne by lqfactor() where the upper
 *     trapezoidal part holds the elementary reflectors.
 *
 * \param[in] tau
 *   The scalar factors of the elementary reflectors.
 *
 * \param[out] W
 *     Workspace matrix,  required size is returned by WorksizeMultQ().
 *
 * \param[in] flags
 *     Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT*, *ARMAS_TRANS*
 *       
 * \param[in,out] conf
 *     Blocking configuration. Field LB defines block sized. If it is zero
 *     unblocked invocation is assumed.
 *
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code
 * Compatible with lapack.DORMLQ
 *
 * #### Notes
 *   m(A) is number of elementary reflectors == A.rows
 *   n(A) is the order of the Q matrix == A.cols
 *
 * \cond
 *   LEFT : m(C) == n(A)
 *   RIGHT: n(C) == n(A)
 * \endcond
 * \ingroup lapack
 */
int armas_x_lqmult(armas_x_dense_t * C,
                   const armas_x_dense_t * A,
                   const armas_x_dense_t * tau, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_x_lqmult_w(C, A, tau, flags, &wb, cf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    int stat = armas_x_lqmult_w(C, A, tau, flags, wbs, cf);
    armas_wrelease(&wb);
    return stat;
}


/**
 * @brief Multiply with orthogonal matrix Q from LQ factorization
 *
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(k)H(k-1)...H(1)
 *
 * as returned by armas_x_lqfactor().
 *
 * @param[in,out] C
 *     On entry, the M-by-N matrix C or if flag bit RIGHT is set then
 *     N-by-M matrix.  On exit C is overwritten by Q*C or Q.T*C.
 *     If bit RIGHT is set then C is  overwritten by C*Q or C*Q.T
 *
 * @param[in] A
 *     LQ factorization as returned by lqfactor() where the upper
 *     trapezoidal part holds the elementary reflectors.
 *
 * @param[in] tau
 *   The scalar factors of the elementary reflectors.
 *
 * @param[in] flags
 *     Indicators. Valid indicators *ARMAS_LEFT*, *ARMAS_RIGHT*, *ARMAS_TRANS*
 *       
 * @param[out] W
 *    Workspace buffer needed for computation. To compute size of the required space call 
 *    the function with workspace bytes set to zero. Size of workspace is returned in 
 *    `wb.bytes` and no other computation or parameter size checking is done and function
 *    returns with success.
 *
 * @param[in,out] conf
 *     Blocking configuration. Field LB defines block sized. If it is zero
 *     unblocked invocation is assumed.
 *
 * @retval  0 Success
 * @retval -1 Error, `conf.error` holds error code
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if m(C) != n(A) for C*op(Q) or n(C) != n(A) for op(Q)*C
 *   - `ARMAS_EINVAL` C or A or tau is null pointer
 *   - `ARMAS_EWORK`  if workspace is less than required for unblocked computation
 *
 * Compatible with lapack.DORMLQ
 *
 * #### Notes
 *   m(A) is number of elementary reflectors == A.rows
 *   n(A) is the order of the Q matrix == A.cols
 *
 * \cond
 *   LEFT : m(C) == n(A)
 *   RIGHT: n(C) == n(A)
 * \endcond
 * \ingroup lapack
 */
int armas_x_lqmult_w(armas_x_dense_t * C,
                     const armas_x_dense_t * A,
                     const armas_x_dense_t * tau,
                     int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t T, Wrk;
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
    // check sizes; A, tau return from armas_x_qrfactor()
    P = (flags & ARMAS_RIGHT) != 0 ? C->cols : C->rows;
    if (P != A->cols || armas_x_size(tau) != A->rows) {
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
        armas_x_make(&Wrk, K, 1, K, buf);
        if ((flags & ARMAS_RIGHT) != 0) {
            unblk_lqmult_right(C, (armas_x_dense_t *) A,
                                 (armas_x_dense_t *) tau, &Wrk, flags, conf);
        } else {
            unblk_lqmult_left(C, (armas_x_dense_t *) A,
                                (armas_x_dense_t *) tau, &Wrk, flags, conf);
        }
    } else {
        // blocked code; block reflector T and temporary space
        armas_x_make(&T, lb, lb, lb, buf);
        armas_x_make(&Wrk, K, lb, K, &buf[armas_x_size(&T)]);

        if ((flags & ARMAS_RIGHT) != 0) {
            blk_lqmult_right(C, (armas_x_dense_t *) A,
                               (armas_x_dense_t *) tau, &T, &Wrk, flags, lb,
                               conf);
        } else {
            blk_lqmult_left(C, (armas_x_dense_t *) A, (armas_x_dense_t *) tau,
                              &T, &Wrk, flags, lb, conf);
        }
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
