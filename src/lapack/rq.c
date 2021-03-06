
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! RQ factorization

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_rqfactor)
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

#define IFERROR(exp) do { \
  int _e = (exp); \
  if (_e) { printf("error at: %s:%d\n", __FILE__, __LINE__); } \
  } while (0);

/*
 * RQ factorization of matrix A.
 *
 *  $$ A = R*Q == \left(0 R right\) \left( Q_1 \over Q_2 \right)  = R*Q_2 $$
 *
 *  where $$ A \in R^{m x n}, R \in R^{m x m}, Q_1 \in R^{m x n} and Q_2 \in R^{{n-m} x n}$$
 *
 * $Q_1$
 */

/*
 * Unblocked factorization.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v**T
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(n-k+i+1:n) = 0 and v(n-k+i) = 1; v(1:n-k+i-1) is stored on exit in
 *  A(m-k+i,1:n-k+i-1), and tau in TAU(i).
 *
 *  m >= n
 *   ( v1 v1 v1 r  r  r  r )
 *   ( v2 v2 v2 v2 r  r  r )
 *   ( v3 v3 v3 v3 v3 r  r )
 *   ( v4 v4 v4 v4 v4 v4 r )
 */
static
int unblk_rqfactor(armas_dense_t * A, armas_dense_t * tau,
                   armas_dense_t * W, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a11, a01, a10, A22;
    armas_dense_t tT, tB, t0, t1, t2, w12;

    EMPTY(ATL);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, 0, ARMAS_PBOTTOM);

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            &a10, &a11, __nil,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        armas_compute_householder(&a11, &a10, &t1, conf);

        armas_make(&w12, a01.rows, 1, a01.rows, armas_data(W));

        armas_apply_householder2x1(&t1, &a10,
                                     &a01, &A00, &w12, ARMAS_RIGHT, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PTOP);
    }
    return 0;
}


/*
 * Blocked factorization.
 */
static
int blk_rqfactor(armas_dense_t * A, armas_dense_t * tau,
                 armas_dense_t * Twork, armas_dense_t * W, int lb,
                 armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A01, A10, A11, A22, AL;
    armas_dense_t tT, tB, t0, t1, t2, w1, Wrk;

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, 0, ARMAS_PBOTTOM);

    while (ATL.rows - lb > 0 && ATL.cols - lb > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            &A10, &A11, __nil,
            __nil, __nil, &A22, /**/ A, lb, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, A11.cols, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        // decompose current panel AL = ( A10 A11 )
        armas_make(&w1, A11.rows, A11.rows, A11.rows, armas_data(W));
        mat_merge1x2(&AL, &A10, &A11);
        unblk_rqfactor(&AL, &t1, &w1, conf);

        // build block reflector
        armas_mscale(Twork, ZERO, 0, conf);
        armas_unblk_rq_reflector(Twork, &AL, &t1, conf);

        // update ( A00 A01 )
        armas_make(&Wrk, A01.rows, A01.cols, A01.rows, armas_data(W));
        armas_update_rq_right(&A01, &A00,
                                &A11, &A10, Twork, &Wrk, FALSE, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PTOP);
    }

    // last block with unblocked
    if (ATL.rows > 0 && ATL.cols > 0) {
        armas_submatrix(&w1, W, 0, 0, ATL.rows, 1);
        unblk_rqfactor(&ATL, &tT, &w1, conf);
    }

    return 0;
}

/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T
 *
 * where  C = ( C1 )   Y = ( Y2 Y1 )
 *            ( C2 )
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb triuu, Y2 is nb*P, T is nb*nb,  W is K*nb
 */
int armas_update_rq_left(armas_dense_t * C1, armas_dense_t * C2,
                           armas_dense_t * Y1, armas_dense_t * Y2,
                           armas_dense_t * T, armas_dense_t * W,
                           int transpose, armas_conf_t * conf)
{
    require(C1->cols == C2->cols && W->rows == C1->cols && W->cols == C1->rows);

    if (armas_size(C1) == 0 && armas_size(C2) == 0)
        return 0;
    // W = C1.T
    armas_mcopy(W, C1, ARMAS_TRANS, conf);
    // W = C1.T*Y1.T = W*Y1.T
    armas_mult_trm(W, ONE, Y1,
                     ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
    // W = W + C2.T*Y2.T
    armas_mult(ONE, W, ONE, C2, Y2, ARMAS_TRANSA | ARMAS_TRANSB, conf);
    // here: W = C.T*Y

    int bits = ARMAS_LOWER | ARMAS_RIGHT;
    if (!transpose)
        bits |= ARMAS_TRANSA;
    // W = W*T or W.T*T
    armas_mult_trm(W, ONE, T, bits, conf);
    // here: W == C.T*Y*T or C.T*Y*T.T

    // C2 = C2 - Y2*W.T
    armas_mult(ONE, C2, -ONE, Y2, W, ARMAS_TRANSA|ARMAS_TRANSB, conf);
    // W = Y1*W.T ==> W.T = W*Y1
    armas_mult_trm(W, ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
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
 * where  C = ( C1 C2 )   Y = ( Y1 Y2 )
 *
 * C1 is K*nb, C2 is K*P, Y1 is nb*nb trilu, Y2 is nb*P, T is nb*nb, W = K*nb
*/
int armas_update_rq_right(armas_dense_t * C1, armas_dense_t * C2,
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
    armas_mult_trm(W, ONE, Y1,
                     ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT|ARMAS_TRANSA, conf);
    // W = W + C2*Y2.T
    armas_mult(ONE, W, ONE, C2, Y2, ARMAS_TRANSB, conf);
    // here: W = C*Y

    int bits = ARMAS_LOWER | ARMAS_RIGHT;
    if (transpose)
        bits |= ARMAS_TRANSA;
    // W = W*T or W.T*T
    armas_mult_trm(W, ONE, T, bits, conf);
    // here: W == C*Y*T or C*Y*T.T

    // C2 = C2 - W*Y2
    armas_mult(ONE, C2, -ONE, W, Y2, ARMAS_NONE, conf);
    // C1 = C1 - W*Y1
    //  W = W*Y1.T
    armas_mult_trm(W, ONE, Y1, ARMAS_LOWER|ARMAS_UNIT|ARMAS_RIGHT, conf);
    // C1 = C1 - W
    armas_mplus(ONE, C1, -ONE, W, ARMAS_NONE, conf);
    // here: C = C*(I - Y*T*Y.T)*C or C = C*(I - Y*T.Y.T).T
    return 0;
}

/*
 * Build block reflector T from elementary reflectors stored in TriLU(A)
 * and coefficients in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  0 |   z = -tau*T*Y.T*v
 *     | z  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 *
 */
int armas_unblk_rq_reflector(armas_dense_t * T, armas_dense_t * A,
                               armas_dense_t * tau, armas_conf_t * conf)
{
    double tauval;
    armas_dense_t ATL, ABR, A00, a10, a11, A20, a21, A22;
    armas_dense_t TTL, TBR, T00, t11, t21, T22;
    armas_dense_t tT, tB, t0, t1, t2;

    EMPTY(ATL);
    EMPTY(A00);
    EMPTY(TTL);
    EMPTY(T00);
    EMPTY(t11);

    mat_partition_2x2(
        &ATL, __nil,
        __nil, &ABR, /**/ A, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x2(
        &TTL, __nil,
        __nil, &TBR, /**/ T, 0, 0, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, 0, ARMAS_PBOTTOM);

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10, &a11, __nil,
            &A20, &a21, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x2to3x3(
            &TTL,
            &T00, __nil, __nil,
            __nil, &t11, __nil,
            __nil, &t21, &T22, /**/ T, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        tauval = armas_get(&t1, 0, 0);
        if (tauval != 0.0) {
            armas_set(&t11, 0, 0, tauval);
            // t21 := -tauval*(a21 + &A20*a10)
            armas_axpby(ZERO, &t21, ONE, &a21, conf);
            armas_mvmult(-tauval, &t21, -tauval, &A20, &a10, ARMAS_NONE,
                           conf);
            // t01 := T22*t21
            armas_mvmult_trm(&t21, ONE, &T22, ARMAS_LOWER, conf);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_3x3to2x2(
            &TTL, __nil,
            __nil, &TBR, /**/ &T00, &t11, &T22, T, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PTOP);
    }
    return 0;
}

/*
 * Build block reflector from RQ factorized matrix.
 *
 * Elementary reflector stored in matrix A rowwise as descriped below. Result
 * block reflector matrix is lower triangular with tau-vector on diagonal.
 *
 *    ( v1 v1 v1 1  .  . )  ( t1 )    ( t1 .  .  )
 *    ( v2 v2 v2 v2 1  . )  ( t2 )    ( t  t2 .  )
 *    ( v3 v3 v3 v3 v3 1 )  ( t3 )    ( t  t  t3 )
 */
int armas_rqreflector(armas_dense_t * T, armas_dense_t * A,
                        armas_dense_t * tau, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (T->cols < A->rows || T->rows < A->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    armas_unblk_rq_reflector(T, A, tau, conf);
    return 0;
}

/**
 * @brief Compute RQ factorization of a M-by-N matrix A
 *
 * @see armas_rqfactor_w
 * @ingroup lapack
 */
int armas_rqfactor(armas_dense_t * A,
                     armas_dense_t * tau, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_rqfactor_w(A, tau, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_rqfactor_w(A, tau, wbs, cf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Compute RQ factorization of a M-by-N matrix A
 *
 * @param[in,out] A
 *    On entry, the M-by-N matrix A, M <= N. On exit, upper triangular matrix R
 *    and the orthogonal matrix Q as product of elementary reflectors.
 *
 * @param[out] tau
 *    On exit, the scalar factors of the elemenentary reflectors.
 *
 * @param[out]  wb
 *    Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *    immediately.
 *
 * @param[in,out] conf
 *    The configuration options.
 *
 * @retval  0 Success
 * @retval <0 Error.
 *
 * Additional information
 *
 * Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *   \f$ Q = H_0 H-1,...,H_{K-1} \f$ , where \f$ K = \min M N \f$
 *
 * Elementary reflector H(k) is stored on first N-M+k elements of row k of A.
 * with implicit unit value on element N-M+k entry. The vector *tau* holds scalar
 * factors of the elementary reflectors.
 *
 * Contents of matrix A after factorization is as follow:
 *```txt
 *      ( v0 v0 r  r  r  r )  M=4, N=6
 *      ( v1 v1 v1 r  r  r )  r  is element of R
 *      ( v2 v2 v2 v2 r  r )  vk is element of H(k)
 *      ( v3 v3 v3 v3 v3 r )
 *```
 * Compatible with lapack.DGERQF
 * @ingroup lapack
 */
int armas_rqfactor_w(armas_dense_t * A,
                       armas_dense_t * tau,
                       armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t T, Wrk;
    armas_env_t *env;
    size_t wsmin, wsneed, wsz = 0;
    int lb;
    DTYPE *buf;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->rows > env->lb)
            wb->bytes = (A->rows * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = A->rows * sizeof(DTYPE);
        return 0;
    }
    // must have: M <= N
    if (A->rows > A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    if (!armas_isvector(tau) || armas_size(tau) != A->rows) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    lb = env->lb;
    wsmin = A->rows * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    // adjust blocking factor for workspace
    wsneed = (lb > 0 ? A->rows * lb : A->rows) * sizeof(DTYPE);
    if (lb > 0 && wsz < wsneed) {
        lb = (wsz / (A->rows * sizeof(DTYPE))) & ~0x3;
        if (lb < ARMAS_BLOCKING_MIN)
            lb = 0;
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || A->rows <= lb) {
        armas_make(&Wrk, A->rows, 1, A->rows, buf);
        unblk_rqfactor(A, tau, &Wrk, conf);
    } else {
        // block reflector [lb,lb]; temp space [n(A)-lb, lb] matrix
        armas_make(&T, lb, lb, lb, buf);
        armas_make(&Wrk, A->rows-lb, lb, A->rows-lb, &buf[armas_size(&T)]);

        blk_rqfactor(A, tau, &T, &Wrk, lb, conf);
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
