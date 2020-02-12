
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate orthogonal Q of RQ factorization

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_rqbuild)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_householder)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
//! \endcond

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(0)H(1)...H(k-1)  , 0 < k < M, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(0)*H(1)...*H(k-1)*I ie. from top to bottom.
 *
 * Compatible to lapack.xORG2R subroutine.
 */
static
int unblk_rqbuild(armas_x_dense_t * A, armas_x_dense_t * tau,
                  armas_x_dense_t * W, int mk, int nk, int mayclear,
                  armas_conf_t * conf)
{
    DTYPE tauval;
    armas_x_dense_t ATL, ABL, ATR, ABR, A00, a01, a10, a11, a12, A22, D;
    armas_x_dense_t tT, tB, t0, t1, t2, w12;


    EMPTY(a11);

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, mk, nk, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, mk, ARMAS_PTOP);

    // zero the top part and to unit matrix
    if (mk > 0 && mayclear) {
        armas_x_mscale(&ATL, ZERO, 0, conf);
        armas_x_mscale(&ATR, ZERO, 0, conf);
        armas_x_diag(&D, &ATL, ATL.cols - mk);
        armas_x_madd(&D, ONE, 0, conf);
    }

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            &a10, &a11, &a12,
            __nil, __nil, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a01), 1);

        armas_x_apply_householder2x1(&t1, &a10,
                                     &a01, &A00, &w12, ARMAS_RIGHT, conf);

        tauval = armas_x_get(&t1, 0, 0);
        armas_x_scale(&a10, -tauval, conf);
        armas_x_set(&a11, 0, 0, 1.0 - tauval);
        armas_x_scale(&a12, ZERO, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            __nil, &ABR, /**/ &A00, &a11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PBOTTOM);
    }
    return 0;
}

/*
 * Blocked code.
 */
static
int blk_rqbuild(armas_x_dense_t * A, armas_x_dense_t * tau,
                armas_x_dense_t * T, armas_x_dense_t * W, int K, int lb,
                armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABL, ABR, ATR, A00, A01, A10, A11, A12, A22, AL, D;
    armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    int mk, nk, uk;

    mk = A->rows - K;
    nk = A->cols - K;
    uk = K % lb;

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, mk + uk, nk + uk, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, mk + uk, ARMAS_PTOP);

    // zero the top part
    if (mk + uk > 0) {
        armas_x_mscale(&ATR, ZERO, 0, conf);
        if (uk > 0) {
            // number of reflector is not multiple of blocking factor
            unblk_rqbuild(&ATL, &tT, W, ATL.rows-uk, ATL.cols-uk, TRUE, conf);
        } else {
            // blocking factor is multiple of K
            armas_x_mscale(&ATL, ZERO, 0, conf);
            armas_x_diag(&D, &ATL, ATL.cols - ATL.rows);
            armas_x_madd(&D, ONE, 0, conf);
        }
    }

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            &A10, &A11, &A12,
            __nil, __nil, &A22, /**/ A, lb, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, A11.cols, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        mat_merge1x2(&AL, &A10, &A11);

        // build block reflector
        armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
        armas_x_mscale(&Tcur, ZERO, 0, conf);
        armas_x_unblk_rq_reflector(&Tcur, &AL, &t1, conf);

        // update A00, A01
        armas_x_submatrix(&Wrk, W, 0, 0, A01.rows, A01.cols);
        armas_x_update_rq_right(&A01, &A00,
                                &A11, &A10, &Tcur, &Wrk, TRUE, conf);

        // update current block
        unblk_rqbuild(&AL, &t1, W, 0, A10.cols, FALSE, conf);

        // zero top rows
        armas_x_mscale(&A12, ZERO, 0, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PBOTTOM);
    }
    return 0;
}

/**
 * \brief Generate the orthogonal Q matrix of RQ factorization
 *
 * Generates the M by N matrix Q with orthonormal rows which
 * are defined as the last M rows of the product of K elementary
 * reflectors of order N.
 *
 *   \f$ Q = H_0 H_1...H_{k-1} , 0 < k < M, H_i = I - tau*v*v^T \f$
 *
 * \param[in,out]  A
 *     On entry, the elementary reflectors as returned by rqfactor().
 *     On exit, the orthogonal matrix Q
 *
 * \param[in]  tau
 *    Scalar coefficents of elementary reflectors
 *
 * \param[out]  W
 *      Workspace
 *
 * \param[in]  K
 *     The number of elementary reflector whose product define the matrix Q
 *
 * \param[in,out] conf
 *     Optional blocking configuration.
 *
 * \retval  0 Succes
 * \retval -1 Failure, `conf.error` holds error code.
 *
 * Compatible with lapackd.ORGRQ.
 */
int armas_x_rqbuild(armas_x_dense_t * A,
                    const armas_x_dense_t * tau, int K, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_x_rqbuild_w(A, tau, K, &wb, cf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    int stat = armas_x_rqbuild_w(A, tau, K, wbs, cf);
    armas_wrelease(&wb);
    return stat;
}


int armas_x_rqbuild_w(armas_x_dense_t * A,
                      const armas_x_dense_t * tau,
                      int K, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t T, Wrk;
    armas_env_t *env;
    size_t wsmin, wsz = 0;
    int lb;
    DTYPE *buf;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->rows > env->lb)
            wb->bytes = (A->rows * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = A->rows * sizeof(DTYPE);
        return 0;
    }

    if (!tau) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    if (armas_x_size(tau) != A->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    lb = env->lb;
    wsmin = A->rows * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // adjust blocking factor for workspace
    //wsneed = (lb > 0 && A->rows > lb ? A->rows * lb : A->rows) * sizeof(DTYPE);
    if (lb > 0 && A->rows > lb) {
        wsz /= sizeof(DTYPE);
        if (wsz < A->rows * lb) {
            lb = (wsz / A->cols) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || A->cols <= lb) {
        armas_x_make(&Wrk, A->rows, 1, A->rows, buf);
        // start row: A->rows - K, column: A.cols - K
        unblk_rqbuild(A, (armas_x_dense_t *) tau, &Wrk, A->rows-K,
                      A->cols-K, TRUE, conf);
    } else {
        // block reflector [lb, lb]; other temporary [m(A)-lb, lb]
        armas_x_make(&T, lb, lb, lb, buf);
        armas_x_make(&Wrk, A->rows-lb, lb, A->rows-lb, &buf[armas_x_size(&T)]);

        blk_rqbuild(A, (armas_x_dense_t *) tau, &T, &Wrk, K, lb, conf);
    }
    armas_wsetpos(wb, wsz);
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
