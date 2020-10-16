
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! QL factorization

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_qlfactor) && defined(armas_qlfactor_w)
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

#ifndef IFERROR
#define IFERROR(exp) do { \
  int _e = (exp); \
  if (_e) { printf("error at: %s:%d\n", __FILE__, __LINE__); } \
  } while (0);
#endif

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif


/*
 *   QL factorization.
 *
 *    A = Q*L = ( Q1 Q2 )*( 0 )  = Q2*L
 *                        ( L )
 *
 *  For M-by-N matrix A the orthogonal matrix Q is product of N elementary
 *  reflectors H(k) = I - tau*v*v.T
 *
 *   H(i)*A = ( H(i)  0 ) * ( A0 ) = ( H(i)*A0 )
 *            (   0   I ) * ( A1 )   (   A1    )
 *
 *   H(i)*A0 = ( I - tau * ( v ) * ( v.T 1 ) ) * ( A0 )
 *             (           ( 1 )             )   ( a1 )
 *
 *           = ( A0 ) - tau * ( v*v.T  v ) ( A0 )
 *             ( a1 )         ( v.T    1 ) ( a1 )
 *
 *           = ( A0 - tau* (v*v.T*A0 + v*a1) )
 *             ( a1 - tau* (v.T*A0 + a1)     )
 *
 *           = ( A0 - tau*v*w )  where w = v.T*A0 + a1 = A0.T*v + a1
 *             ( a1 - tau*w   )
 */


/*
 * Unblocked factorization.
 */
static
int unblk_qlfactor(armas_dense_t * A, armas_dense_t * tau,
                   armas_dense_t * W, armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, a01, a10, a11, A22;
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
        armas_compute_householder(&a11, &a01, &t1, conf);

        armas_make(&w12, a10.cols, 1, a10.cols, armas_data(W));

        armas_apply_householder2x1(&t1, &a01,
                                     &a10, &A00, &w12, ARMAS_LEFT, conf);
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
int blk_qlfactor(armas_dense_t * A, armas_dense_t * tau,
                 armas_dense_t * T, armas_dense_t * W, int lb,
                 armas_conf_t * conf)
{
    armas_dense_t ATL, ABR, A00, A01, A10, A11, A22, AT;
    armas_dense_t tT, tB, t0, t1, t2, w12, Wrk;

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
            &tT, &t0, &t1, &t2, /**/ tau, lb, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        // current panel ( A01 )
        //               ( A11 )
        armas_make(&w12, A11.cols, 1, A11.cols, armas_data(W));
        mat_merge2x1(&AT, &A01, &A11);
        unblk_qlfactor(&AT, &t1, &w12, conf);

        // build reflector T
        armas_mscale(T, ZERO, 0, conf);
        armas_unblk_ql_reflector(T, &AT, &t1, conf);

        // update with (I - Y*T*Y.T).T
        armas_make(&Wrk, A10.cols, A10.rows, A10.cols, armas_data(W));
        armas_update_ql_left(&A10, &A00, &A11, &A01, T, &Wrk, TRUE, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PTOP);
    }

    // last block with unblocked
    if (ATL.rows > 0 && ATL.cols > 0) {
        armas_submatrix(&w12, W, 0, 0, ATL.cols, 1);
        unblk_qlfactor(&ATL, &t0, &w12, conf);
    }

    return 0;
}

/*
 * compute:
 *      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T
 * or
 *      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T
 *
 * where  C = ( C2 )   Y = ( Y2 )
 *            ( C1 )       ( Y1 )
 *
 * C1 is nb*K, C2 is P*K, Y1 is nb*nb triuu, Y2 is P*nb, T is nb*nb,  W is K*nb
 */
int armas_update_ql_left(armas_dense_t * C1, armas_dense_t * C2,
                     armas_dense_t * Y1, armas_dense_t * Y2,
                     armas_dense_t * T, armas_dense_t * W, int transpose,
                     armas_conf_t * conf)
{
    require(C1->cols == C2->cols && W->rows == C1->cols && W->cols == C1->rows);

    if (armas_size(C1) == 0 && armas_size(C2) == 0)
        return 0;
    // W = C1.T
    armas_mcopy(W, C1, ARMAS_TRANS, conf);
    // W = C1.T*Y1 = W*Y1
    armas_mult_trm(W, ONE, Y1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf);
    // W = W + C2.T*Y2
    armas_mult(ONE, W, ONE, C2, Y2, ARMAS_TRANSA, conf);
    // here: W = C.T*Y

    int bits = ARMAS_LOWER | ARMAS_RIGHT;
    if (!transpose)
        bits |= ARMAS_TRANSA;
    // W = W*T or W.T*T
    armas_mult_trm(W, ONE, T, bits, conf);
    // here: W == C.T*Y*T or C.T*Y*T.T

    // C2 = C2 - Y2*W.T
    armas_mult(ONE, C2, -ONE, Y2, W, ARMAS_TRANSB, conf);
    // W = Y1*W.T ==> W.T = W*Y1.T
    armas_mult_trm(W, ONE, Y1,
                     ARMAS_UPPER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf);
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
 * where  C = ( C2 C1 )   Y = ( Y2 )
 *                            ( Y1 )
 *
 * C1 is K*nb, C2 is K*P, Y1 is nb*nb triuu, Y2 is P*nb, T is nb*nb, W = K*nb
*/
int armas_update_ql_right(armas_dense_t * C1, armas_dense_t * C2,
                      armas_dense_t * Y1, armas_dense_t * Y2,
                      armas_dense_t * T, armas_dense_t * W, int transpose,
                      armas_conf_t * conf)
{
   require(C1->rows == C2->rows && W->rows == C1->rows && W->cols == C1->cols);

    if (armas_size(C1) == 0 && armas_size(C2) == 0)
        return 0;
    // W = C1
    armas_mcopy(W, C1, 0, conf);
    // W = C1*Y1 = W*Y1
    armas_mult_trm(W, ONE, Y1, ARMAS_UPPER|ARMAS_UNIT|ARMAS_RIGHT, conf);
    // W = W + C2*Y2
    armas_mult(ONE, W, ONE, C2, Y2, ARMAS_NONE, conf);
    // here: W = C*Y

    int bits = ARMAS_LOWER | ARMAS_RIGHT;
    if (transpose)
        bits |= ARMAS_TRANSA;
    // W = W*T or W.T*T
    armas_mult_trm(W, ONE, T, bits, conf);
    // here: W == C*Y*T or C*Y*T.T

    // C2 = C2 - W*Y2.T
    armas_mult(ONE, C2, -ONE, W, Y2, ARMAS_TRANSB, conf);
    // C1 = C1 - W*Y1*T
    //  W = W*Y1.T
    armas_mult_trm(W, ONE, Y1,
                     ARMAS_UPPER|ARMAS_UNIT|ARMAS_TRANSA|ARMAS_RIGHT, conf);
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
 * T = | T  0 |   z = -tau*T*Y.T*v
 *     | z  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 */
int armas_unblk_ql_reflector(armas_dense_t * T, armas_dense_t * A,
                               armas_dense_t * tau, armas_conf_t * conf)
{
    double tauval;
    armas_dense_t ATL, ABR, A00, a01, A02, a11, a12, A22;
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
            &A00, &a01, &A02,
            __nil, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x2to3x3(
            &TTL,
            &T00, __nil, __nil,
            __nil, &t11, __nil,
            __nil, &t21, &T22, /**/ T, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        tauval = armas_get(&t1, 0, 0);
        if (tauval != ZERO) {
            armas_set(&t11, 0, 0, tauval);
            // t21 := -tauval*(a12.T + &A02.T*a01)
            armas_axpby(ZERO, &t21, ONE, &a12, conf);
            armas_mvmult(-tauval, &t21, -tauval, &A02, &a01, ARMAS_TRANSA,
                           conf);
            // t21 := T22*t21
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

/**
 * @brief Build the QL block reflector
 *
 * @param[out] T
 *   On exit the requested block reflector.
 * @param[in] A
 *   The QL factored matrix
 * @param[in] tau
 *   The factorization scalar values/
 * @param[in,out] conf
 *   Configuration block.
 *
 * @retval  0 Success
 * @retval <0 Failure
 * @ingroup lapack
 */
int armas_qlreflector(armas_dense_t * T, armas_dense_t * A,
                        armas_dense_t * tau, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (T->cols < A->cols || T->rows < A->cols) {
        conf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    armas_unblk_ql_reflector(T, A, tau, conf);
    return 0;
}

/**
 * @brief Compute QL factorization of a M-by-N matrix A
 *
 * @see armas_qlfactor_w
 * @ingroup lapack
 */
int armas_qlfactor(armas_dense_t * A,
                     armas_dense_t * tau, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_qlfactor_w(A, tau, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_qlfactor_w(A, tau, wbs, cf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Compute QL factorization of a M-by-N matrix A
 *
 * @param[in,out] A
 *    On entry, the M-by-N matrix A, M >= N. On exit, lower triangular matrix L
 *    and the orthogonal matrix Q as product of elementary reflectors.
 *
 * @param[out] tau
 *   Vector of length N. On exit, the scalar factors of the elemenentary reflectors.
 *
 * @param wb
 *   Workspace. If *wb.bytes* is zero then size of required workspace in computed and returned
 *   immediately.
 *
 * @param[in,out] conf
 *   The blocking configuration. If nil then default blocking configuration
 *
 * @retval  0 Success
 * @retval <0 Failure, conf.error holds error code
 *
 *  Last error codes returned
 *   - `ARMAS_ESIZE`  if M < N 
 *   - `ARMAS_EINVAL` tau is not column vector or len(tau) < N
 *   - `ARMAS_EWORK`  if workspace is less than N elements
 *
 * Additional information
 *
 * Ortogonal matrix Q is product of elementary reflectors H(k)
 *
 *    \f$ Q = H_{k-1}...H_1 H_0, where K = min(M,N) \f$
 *
 *  Elementary reflector H(k) is stored on column k of A above the diagonal with
 *  implicit unit value on diagonal entry. The vector TAU holds scalar factors
 *  of the elementary reflectors.
 *
 *  Contents of matrix A after factorization is as follow:
 *```txt
 *      ( v0 v1 v2 v3 )   for M=6, N=4
 *      ( v0 v1 v2 v3 )   l is element of L
 *      ( l  v1 v2 v3 )   vk is element of H(k)
 *      ( l  l  v2 v3 )
 *      ( l  l  l  v3 )
 *      ( l  l  l  l  )
 *```
 *  armas_qlfactor_w() is compatible with lapack.DGEQLF
 * @ingroup lapack
 */
int armas_qlfactor_w(armas_dense_t * A,
                       armas_dense_t * tau,
                       armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t T, Wrk;
    armas_env_t *env;
    size_t wsmin, wsz = 0;
    DTYPE *buf;
    int lb;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->cols > env->lb)
            wb->bytes = (env->lb * A->cols) * sizeof(DTYPE);
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
            lb = (wsz / A->cols) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || A->cols <= lb) {
        armas_make(&Wrk, A->cols, 1, A->cols, buf);
        unblk_qlfactor(A, tau, &Wrk, conf);
    } else {
        // block reflector [lb, lb]; temporary space [N(A)-lb,lb] matrix
        armas_make(&T, lb, lb, lb, buf);
        armas_make(&Wrk, A->cols-lb, lb, A->cols-lb, &buf[armas_size(&T)]);

        blk_qlfactor(A, tau, &T, &Wrk, lb, conf);
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defined. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
