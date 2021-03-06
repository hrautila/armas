
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! QR factorization


#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_qrfactor) && defined(armas_qrfactor_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_householder)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include <assert.h>
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


/*
 * Unblocked factorization.
 */
static
int armas_unblk_qrfactor(armas_dense_t * A, armas_dense_t * tau,
                           armas_dense_t * W, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a11, a12, a21, A22;
    armas_dense_t tT, tB, t0, t1, t2, w12;

    // initialize to "empty" to avoid "maybe-uninitialized" errors
    EMPTY(ATL);
    EMPTY(ABR);
    EMPTY(A00);
    EMPTY(a11);
    EMPTY(A22);

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        armas_compute_householder(&a11, &a21, &t1, conf);

        armas_make(&w12, a12.cols, 1, a12.cols, armas_data(W));

        armas_apply_householder2x1(&t1, &a21, &a12, &A22,
                                     &w12, ARMAS_LEFT, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PBOTTOM);
    }
    return 0;
}


/*
 * Blocked factorization.
 */
static
int blk_qrfactor(armas_dense_t * A, armas_dense_t * tau,
                 armas_dense_t * Twork, armas_dense_t * W, int lb,
                 armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A11, A12, A21, A22, AL;
    armas_dense_t tT, tB, t0, t1, t2, w1, Wrk;
    DTYPE *wdata = armas_data(W); 
    // initialize to "empty" to avoid "maybe-uninitialized" errors
    EMPTY(ATL);
    EMPTY(ABR);
    EMPTY(A00);
    EMPTY(A11);
    EMPTY(A22);

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, 0, ARMAS_PTOP);

    while (ABR.rows - lb > 0 && ABR.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            __nil, &A11, &A12,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, lb, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        // decompose current panel AL = ( A11 )
        //                              ( A21 )
        armas_make(&w1, A11.rows, 1, A11.rows, wdata);
        mat_merge2x1(&AL, &A11, &A21);
        armas_unblk_qrfactor(&AL, &t1, &w1, conf);

        // build block reflector
        armas_mscale(Twork, ZERO, 0, conf);
        armas_unblk_qr_reflector(Twork, &AL, &t1, conf);

        // update ( A12 A22 ).T
        armas_make(&Wrk, A12.cols, A12.rows, A12.cols, wdata);
        armas_update_qr_left(&A12, &A22, &A11, &A21, Twork, &Wrk, TRUE, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PBOTTOM);
    }

    // last block with unblocked
    if (ABR.rows > 0 && ABR.cols > 0) {
        armas_submatrix(&w1, W, 0, 0, ABR.cols, 1);
        armas_unblk_qrfactor(&ABR, &t2, &w1, conf);
    }

    return 0;
}

/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T
 *
 * where  C = ( C1 )   Y = ( Y1 )
 *            ( C2 )       ( Y2 )
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb trilu, Y2 is P*nb, T is nb*nb,  W is K*nb
 */
int armas_update_qr_left(armas_dense_t * C1, armas_dense_t * C2,
                           armas_dense_t * Y1, armas_dense_t * Y2,
                           armas_dense_t * T, armas_dense_t * W,
                           int transpose, armas_conf_t * conf)
{
    require(C1->cols == C2->cols && W->rows == C1->cols && W->cols == C1->rows);
    if (armas_size(C1) == 0 && armas_size(C2) == 0)
        return 0;
    // W = C1.T
    armas_mcopy(W, C1, ARMAS_TRANS, conf);
    // W = C1.T*Y1 = W*Y1
    armas_mult_trm (W, ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
    // W = W + C2.T*Y2
    armas_mult(ONE, W, ONE, C2, Y2, ARMAS_TRANSA, conf);
    // here: W = C.T*Y

    int bits = ARMAS_UPPER | ARMAS_RIGHT;
    if (!transpose)
        bits |= ARMAS_TRANSA;
    // W = W*T or W.T*T
    armas_mult_trm(W, ONE, T, bits, conf);
    // here: W == C.T*Y*T or C.T*Y*T.T

    // C2 = C2 - Y2*W.T
    armas_mult(ONE, C2, -ONE, Y2, W, ARMAS_TRANSB, conf);
    // W = Y1*W.T ==> W.T = W*Y1.T
    armas_mult_trm (W, ONE, Y1,
                      ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf);
    // C1 = C1 - W.T
    armas_mplus(ONE, C1, -ONE, W, ARMAS_TRANSB, conf);
    // here: C = (I - Y*T*Y.T)*C or C = (I - Y*T.Y.T).T*C
    return 0;
}


/*
 * compute:
 *      C*Q.T = C*(I -Y*T*Y.T).T ==  C - C*Y*T.T*Y.T
 * or
 *      C*Q   = (I -Y*T*Y.T)*C   ==  C - C*Y*T*Y.T
 *
 * where  C = ( C1 C2 )   Y = ( Y1 )
 *                            ( Y2 )
 *
 * C1 is K*nb, C2 is K*P, Y1 is nb*nb trilu, Y2 is P*nb, T is nb*nb, W = K*nb
*/
int armas_update_qr_right(armas_dense_t * C1, armas_dense_t * C2,
                            armas_dense_t * Y1, armas_dense_t * Y2,
                            armas_dense_t * T, armas_dense_t * W,
                            int transpose, armas_conf_t * conf)
{
    require(C1->rows == C2->rows && W->rows == C1->rows && W->cols == C1->cols);
    if (armas_size(C1) == 0 && armas_size(C2) == 0)
        return 0;
    // W = C1
    armas_mcopy(W, C1, 0, conf);
    // W = C1*Y1 = W*Y1
    armas_mult_trm (W, ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
    // W = W + C2*Y2
    armas_mult(ONE, W, ONE, C2, Y2, ARMAS_NONE, conf);
    // here: W = C*Y

    int bits = ARMAS_UPPER|ARMAS_RIGHT;
    if (transpose)
        bits |= ARMAS_TRANSA;
    // W = W*T or W.T*T
    armas_mult_trm(W, ONE, T, bits, conf);
    // here: W == C*Y*T or C*Y*T.T

    // C2 = C2 - W*Y2.T
    armas_mult(ONE, C2, -ONE, W, Y2, ARMAS_TRANSB, conf);
    // C1 = C1 - W*Y1*T
    //  W = W*Y1.T
    armas_mult_trm (W, ONE, Y1,
                      ARMAS_LOWER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf);
    // C1 = C1 - W
    armas_mplus(ONE, C1, -ONE, W, ARMAS_NONE, conf);
    // here: C = C*(I - Y*T*Y.T)*C or C = C*(I - Y*T.Y.T).T
    return 0;
}

/*
 * Build block reflector T from HH reflector stored in TriLU(A) and coefficients
 * in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  z |   z = -tau*T*Y.T*v
 *     | 0  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 */
int armas_unblk_qr_reflector(armas_dense_t * T, armas_dense_t * A,
                               armas_dense_t * tau, armas_conf_t * conf)
{
    DTYPE tauval;
    armas_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
    armas_dense_t TTL, TBR, T00, t01, T02, t11, t12, T22;
    armas_dense_t tT, tB, t0, t1, t2;

    // initialize to "empty" to avoid "maybe-uninitialized" errors
    EMPTY(ATL);
    EMPTY(ABR);
    EMPTY(A00);
    EMPTY(a11);
    EMPTY(A22);
    EMPTY(TTL);
    EMPTY(TBR);
    EMPTY(T00);
    EMPTY(t11);
    EMPTY(T22);

    mat_partition_2x2(
        &ATL, __nil, __nil, &ABR, /**/ A, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x2(
        &TTL, __nil, __nil, &TBR, /**/ T, 0, 0, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, 0, ARMAS_PTOP);

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10, &a11, __nil,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x2to3x3(
            &TTL,
            &T00, &t01, &T02,
            __nil, &t11, &t12,
            __nil, __nil, &T22, /**/ T, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        tauval = armas_get(&t1, 0, 0);
        if (tauval != 0.0) {
            armas_set(&t11, 0, 0, tauval);
            // t01 := -tauval*(a10.T + &A20.T*a21)
            armas_axpby(ZERO, &t01, ONE, &a10, conf);
            armas_mvmult(-tauval, &t01, -tauval, &A20, &a21, ARMAS_TRANSA,
                           conf);
            // t01 := T00*t01
            armas_mvmult_trm(&t01, ONE, &T00, ARMAS_UPPER, conf);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x3to2x2(
            &TTL, __nil,
            __nil, &TBR, /**/ &T00, &t11, &T22, T, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PBOTTOM);
    }
    return 0;
}

/**
 * @brief Build the QR block reflector
 *
 * @param[out] T
 *   On exit the requested block reflector.
 * @param[in] A
 *   The QR factored matrix
 * @param[in] tau
 *   The factorization scalar values/
 * @param[in,out] conf
 *   Configuration block.
 *
 * @retval  0 Success
 * @retval <0 Failure
 * @ingroup lapack
 */
int armas_qrreflector(armas_dense_t * T, armas_dense_t * A,
                        armas_dense_t * tau, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (T->cols < A->cols || T->rows < A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    armas_unblk_qr_reflector(T, A, tau, conf);
    return 0;
}

/**
 * @brief Compute QR factorization of a M-by-N matrix \f$ A = QR \f$
 *
 * @see armas_qrfactor_w
 * @ingroup lapack
 */
int armas_qrfactor(armas_dense_t * A,
                     armas_dense_t * tau, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_qrfactor_w(A, tau, &wb, conf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_qrfactor_w(A, tau, wbs, conf);
    armas_wrelease(&wb);
    return err;
}


/**
 * @brief Compute QR factorization of a M-by-N matrix \f$ A = QR \f$
 *
 * @param[in,out] A
 *    On entry, the M-by-N matrix A, M >= N. On exit, upper triangular
 *    matrix R and the orthogonal matrix Q as product of elementary
 *    reflectors.
 *
 * @param[out] tau
 *    On exit, the scalar factors of the elementary reflectors.
 *
 * @param[in] wb
 *    Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *    immediately.
 *
 * @param[in,out] conf
 *    The configuration options.
 *
 * @retval  0 Success
 * @retval <0 Failure, `conf.error` set to last error.
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if M < N
 *   - `ARMAS_EINVAL` tau is not column vector or size(tau) < N
 *   - `ARMAS_EWORK`  if workspace is less than N elements
 *
 * Additional information
 *
 *  Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *    \f$  Q = H(0)H(1),...,H(K-1), where K = min(M,N) \f$
 *
 *  Elementary reflector H(k) is stored on column k of A below the diagonal with
 *  implicit unit value on diagonal entry. The vector `tau` holds scalar factors
 *  of the elementary reflectors.
 *
 *  Contents of matrix A after factorization is as follow:
 *```txt
 *      ( r  r  r  r )  M=6, N=4
 *      ( v1 r  r  r )  r   in R
 *      ( v1 v2 r  r )  vk  in H(k)
 *      ( v1 v2 v3 r )
 *      ( v1 v2 v3 v4)
 *      ( v1 v2 v3 v4)
 *```
 *  Compatible with lapack.xGEQRF
 * @ingroup lapack
 */
int armas_qrfactor_w(armas_dense_t * A,
                       armas_dense_t * tau,
                       armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t T, Wrk;
    size_t wsmin, wsz = 0;
    int lb;
    DTYPE *buf;
    armas_env_t *env;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        // if column equal to or less than blocking size; then unblocked size
        if (env->lb > 0 && A->cols > env->lb)
            wb->bytes = (A->cols * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = A->cols * sizeof(DTYPE);
        return 0;
    }
    // must have: M >= N
    if (A->rows < A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    if (!armas_isvector(tau) || armas_size(tau) < A->cols) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    lb = env->lb;
    wsmin = A->cols * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    // adjust blocking factor for workspace
    if (lb > 0 && A->cols > lb) {
        wsz /= sizeof(DTYPE);
        if (wsz < A->cols * lb) {
            // need N*lb elements
            lb = (wsz / A->cols) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || A->cols <= lb) {
        armas_make(&Wrk, A->cols, 1, A->cols, buf);
        armas_unblk_qrfactor(A, tau, &Wrk, conf);
    } else {
        // block reflector [lb,lb]; temporary space [N(A)-lb,lb]
        armas_make(&T, lb, lb, lb, buf);
        armas_make(&Wrk, A->cols - lb, lb, A->cols - lb,
                     &buf[armas_size(&T)]);

        blk_qrfactor(A, tau, &T, &Wrk, lb, conf);
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
