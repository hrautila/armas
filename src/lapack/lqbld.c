
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate orthogonal Q matrix of LQ factorization

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_lqbuild) && defined(armas_x_lqbuild_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_householder)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#ifndef ARMAS_BLOCKING_MIN
#define ARMAS_BLOCKING_MIN 32
#endif

/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(k)H(k-1)...H(1)  , 0 < k <= M, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(k)*H(k-1)...*H(1)*I ie. from bottom to top.
 *
 * If k < M rows k+1:M are cleared and diagonal entries [k+1:M,k+1:M] are
 * set to unit. Then the matrix Q is generated by right multiplying elements below
 * of i'th elementary reflector H(i).
 * 
 * Compatible to lapack.xORG2L subroutine.
 */
static
int unblk_lqbuild(armas_x_dense_t * A, armas_x_dense_t * tau,
                  armas_x_dense_t * W, int mk, int nk, int mayclear,
                  armas_conf_t * conf)
{
    DTYPE tauval;
    armas_x_dense_t ATL, ABL, ABR, A00, a10, a11, a12, a21, A22, D;
    armas_x_dense_t tT, tB, t0, t1, t2, w12;

    EMPTY(ATL);
    EMPTY(A00);
    EMPTY(a11);

    mat_partition_2x2(
        &ATL, __nil,
        &ABL, &ABR, /**/ A, mk, nk, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, mk, ARMAS_PBOTTOM);

    // zero the bottom part
    if (mk > 0 && mayclear) {
        armas_x_mscale(&ABL, ZERO, 0, conf);
        armas_x_mscale(&ABR, ZERO, 0, conf);
        armas_x_diag(&D, &ABR, 0);
        armas_x_madd(&D, ONE, 0, conf);
    }

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &a10, &a11, &a12,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        armas_x_submatrix(&w12, W, 0, 0, armas_x_size(&a21), 1);

        armas_x_apply_householder2x1(&t1, &a12,
                                     &a21, &A22, &w12, ARMAS_RIGHT, conf);

        tauval = armas_x_get(&t1, 0, 0);
        armas_x_scale(&a12, -tauval, conf);
        armas_x_set(&a11, 0, 0, 1.0 - tauval);

        // zero
        armas_x_scale(&a10, ZERO, conf);
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
 * Blocked code.
 */
static
int blk_lqbuild(armas_x_dense_t * A, armas_x_dense_t * tau,
                armas_x_dense_t * T, armas_x_dense_t * W, int K, int lb,
                armas_conf_t * conf)
{
    armas_x_dense_t ATL, ABL, ABR, A00, A10, A11, A12, A21, A22, AL, D;
    armas_x_dense_t tT, tB, t0, t1, t2, Tcur, Wrk;
    int mk, nk, uk;

    EMPTY(ATL);
    EMPTY(A00);

    mk = A->rows - K;
    nk = A->cols - K;
    uk = K % lb;

    mat_partition_2x2(
        &ATL, __nil,
        &ABL, &ABR, /**/ A, mk + uk, nk + uk, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, mk + uk, ARMAS_PBOTTOM);

    // zero the bottom part
    if (mk + uk > 0) {
        armas_x_mscale(&ABL, ZERO, 0, conf);
        if (uk > 0) {
            // number of reflector is not multiple of blocking factor
            unblk_lqbuild(&ABR, &tB, W, ABR.rows-uk, ABR.cols-uk, TRUE, conf);
        } else {
            // blocking factor is multiple of K
            armas_x_mscale(&ABR, ZERO, 0, conf);
            armas_x_diag(&D, &ABR, 0);
            armas_x_madd(&D, ONE, 0, conf);
        }
    }

    while (ATL.rows > 0 && ATL.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, __nil, __nil,
            &A10, &A11, &A12,
            __nil, &A21, &A22, /**/ A, lb, ARMAS_PTOPLEFT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, lb, ARMAS_PTOP);
        // ---------------------------------------------------------------------
        mat_merge1x2(&AL, &A11, &A12);

        // build block reflector
        armas_x_submatrix(&Tcur, T, 0, 0, A11.cols, A11.cols);
        armas_x_mscale(&Tcur, ZERO, 0, conf);
        armas_x_unblk_lq_reflector(&Tcur, &AL, &t1, conf);

        // update A21, A22
        armas_x_submatrix(&Wrk, W, 0, 0, A21.rows, A21.cols);
        armas_x_update_lq_right(&A21, &A22,
                                &A11, &A12, &Tcur, &Wrk, FALSE, conf);

        // update current block
        unblk_lqbuild(&AL, &t1, W, 0, A12.cols, FALSE, conf);

        // zero top rows
        armas_x_mscale(&A10, ZERO, 0, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, __nil,
            &ABL, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PTOPLEFT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PTOP);
    }
    return 0;
}

/**
 * \brief Generate orthogonal Q matrix of LQ factorization
 *
 * Generate the M by N matrix Q with orthogonal rows which
 * are defined as the first M rows of the product of K first elementary
 * reflectors.
 *
 * \param[in,out]  A
 *     On entry, the elementary reflectors as returned by DecomposeLQ().
 *     stored right of diagonal of the M by N matrix A.
 *     On exit, the orthogonal matrix Q
 *
 * \param[in]  tau
 *     Scalar coefficents of elementary reflectors
 *
 * \param[in]  K
 *     The number of elementary reflector whose product define the matrix Q
 *
 * \param[in,out]  conf
 *     Optional blocking configuration.
 *
 * Compatible with lapackd.ORGLQ.
 * \ingroup lapack
 */
int armas_x_lqbuild(armas_x_dense_t * A,
                    const armas_x_dense_t * tau, int K, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_x_lqbuild_w(A, tau, K, &wb, cf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    int stat = armas_x_lqbuild_w(A, tau, K, wbs, cf);
    armas_wrelease(&wb);
    return stat;
}



/**
 * @brief Generate orthogonal Q matrix of LQ factorization
 *
 * Generate the M by N matrix Q with orthogonal rows which
 * are defined as the first M rows of the product of K first elementary
 * reflectors.
 *
 * @param[in,out]  A
 *     On entry, the elementary reflectors as returned by DecomposeLQ().
 *     stored right of diagonal of the M by N matrix A.
 *     On exit, the orthogonal matrix Q
 *
 * @param[in]  tau
 *     Scalar coefficents of elementary reflectors
 *
 * @param[out]  wb
 *    Workspace buffer needed for computation. To compute size of the required space call 
 *    the function with workspace bytes set to zero. Size of workspace is returned in 
 *    `wb.bytes` and no other computation or parameter size checking is done and function
 *    returns with success.
 *
 * @param[in]  K
 *     The number of elementary reflector whose product define the matrix Q
 *
 * @param[in,out]  conf
 *     Optional blocking configuration.
 *
 *  @retval 0  success
 *  @retval -1 error and `conf.error` set to last error
 *
 *  Last error codes returned
 *   - `ARMAS_EINVAL` A or tau is null pointer
 *   - `ARMAS_EWORK`  if no workspace or it is less than required for unblocked computation
 *
 * Compatible with lapackd.ORGLQ.
 * @ingroup lapack
 */
int armas_x_lqbuild_w(armas_x_dense_t * A,
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

    lb = env->lb;
    wsmin = A->rows * sizeof(DTYPE);
    if (!wb || (wsz = armas_wbytes(wb)) < wsmin) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // adjust blocking factor for workspace
    if (lb > 0 && A->rows > lb) {
        wsz /= sizeof(DTYPE);
        if (wsz < A->rows * lb) {
            lb = (wsz / A->rows) & ~0x3;
            if (lb < ARMAS_BLOCKING_MIN)
                lb = 0;
        }
    }

    wsz = armas_wpos(wb);
    buf = (DTYPE *) armas_wptr(wb);

    if (lb == 0 || A->cols <= lb) {
        armas_x_make(&Wrk, A->rows, 1, A->rows, buf);
        unblk_lqbuild(A, (armas_x_dense_t *) tau, &Wrk, A->rows - K,
                        A->cols - K, TRUE, conf);
    } else {
        // block reflector [lb,lb]; temporary space [M(A)-lb,lb] matrix
        armas_x_make(&T, lb, lb, lb, buf);
        armas_x_make(&Wrk, A->rows-lb, lb, A->rows-lb, &buf[armas_x_size(&T)]);

        blk_lqbuild(A, (armas_x_dense_t *) tau, &T, &Wrk, K, lb, conf);
    }
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
