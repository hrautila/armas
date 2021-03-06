
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Generate orthogonal matrix Q of QL factorization

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_qlbuild) && defined(armas_qlbuild_w)
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


/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * Parameter nk is last nk elementary reflectors that are not used in computing
 * the matrix Q. Parameter mk length of the first unused elementary reflectors
 * First nk columns are zeroed and subdiagonal mk-nk is set to unit.
 *
 * Compatible with lapack.DORG2L subroutine.
 */
static
int unblk_qlbuild(armas_dense_t * A, armas_dense_t * tau,
                  armas_dense_t * W, int mk, int nk, int mayclear,
                  armas_conf_t * conf)
{
    armas_dense_t ATL, ABL, ATR, ABR, A00, a01, a10, a11, a21, A22;
    armas_dense_t tT, tB, t0, t1, t2, w12, D;
    DTYPE tauval;

    EMPTY(a11);

    // (mk, nk) = (rows, columns) of upper left partition
    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, mk, nk, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, nk, ARMAS_PTOP);

    // zero the left side
    if (nk > 0 && mayclear) {
        armas_mscale(&ABL, ZERO, 0, conf);
        armas_mscale(&ATL, ZERO, 0, conf);
        armas_diag(&D, &ATL, nk - mk);
        armas_madd(&D, ONE, 0, conf);
    }

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &a01, __nil,
            &a10, &a11, __nil,
            __nil, &a21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        armas_make(&w12, a10.cols, 1, a10.cols, armas_data(W));
        armas_apply_householder2x1(&t1, &a01,
                                     &a10, &A00, &w12, ARMAS_LEFT, conf);

        tauval = armas_get(&t1, 0, 0);
        armas_scale(&a01, -tauval, conf);
        armas_set(&a11, 0, 0, ONE - tauval);

        // zero bottom elements
        armas_scale(&a21, ZERO, conf);
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
 * Blocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * If the number K of elementary reflectors is not multiple of the blocking
 * factor lb, then unblocked code is used first to generate the upper left corner
 * of the matrix Q. 
 *
 * Compatible with lapack.DORGQL subroutine.
 */
static
int blk_qlbuild(armas_dense_t * A, armas_dense_t * tau,
                armas_dense_t * T, armas_dense_t * W, int K, int lb,
                armas_conf_t * conf)
{
    armas_dense_t ATL, ABL, ATR, ABR, A00, A01, A10, A11, A21, A22, AT;
    armas_dense_t tT, tB, t0, t1, t2, D, Tcur, Wrk;
    int mk, nk, uk;

    nk = A->cols - K;
    mk = A->rows - K;
    uk = K % lb;

    // (mk, nk) = (rows, columns) of upper left partition
    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, mk + uk, nk + uk, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &tT, &tB, /**/ tau, nk + uk, ARMAS_PTOP);

    // zero the left side
    if (nk + uk > 0) {
        armas_mscale(&ABL, ZERO, 0, conf);
        if (uk > 0) {
            // number of reflectors is not multiple of blocking factor
            // do the first part with unblocked code.
            unblk_qlbuild(&ATL, &tT, W, ATL.rows-uk, ATL.cols-uk, TRUE, conf);
        } else {
            // blocking factor is multiple of K
            armas_mscale(&ATL, ZERO, 0, conf);
            armas_diag(&D, &ATL, nk - mk);
            armas_madd(&D, ONE, 0, conf);
        }
    }

    while (ABR.rows > 0 && ABR.cols > 0) {
        mat_repartition_2x2to3x3(
            &ATL,
            &A00, &A01, __nil,
            &A10, &A11, __nil,
            __nil, &A21, &A22, /**/ A, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_2x1to3x1(
            &tT, &t0, &t1, &t2, /**/ tau, 1, ARMAS_PBOTTOM);
        // ---------------------------------------------------------------------
        mat_merge2x1(&AT, &A01, &A11);

        // build block reflector
        armas_make(&Tcur, A11.rows, A11.cols, A11.rows, armas_data(T));
        armas_unblk_ql_reflector(&Tcur, &AT, &t1, conf);

        // update left side i.e A00 and A00 with (I - Y*T*Y.T)
        armas_make(&Wrk, A10.cols, A10.rows, A10.cols, armas_data(W));
        armas_update_ql_left(&A10, &A00,
                               &A11, &A01, &Tcur, &Wrk, FALSE, conf);

        // update current block
        unblk_qlbuild(&AT, &t1, W, A01.rows, 0, FALSE, conf);

        // zero bottom rows
        armas_mscale(&A21, ZERO, 0, conf);
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &ATL, &ATR,
            __nil, &ABR, /**/ &A00, &A11, &A22, A, ARMAS_PBOTTOMRIGHT);
        mat_continue_3x1to2x1(
            &tT, &tB, /**/ &t0, &t1, tau, ARMAS_PBOTTOM);
    }
    return 0;
}


/**
 * @brief Generate orthogonal Q matrix of QL factorization
 *
 * @see armas_qlbuild_w
 * @ingroup lapack
 */
int armas_qlbuild(armas_dense_t * A,
                    const armas_dense_t * tau, int K, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if ((err = armas_qlbuild_w(A, tau, K, &wb, cf)) < 0)
        return err;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            cf->error = ARMAS_EMEMORY;
            return -ARMAS_EMEMORY;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_qlbuild_w(A, tau, K, wbs, cf);
    armas_wrelease(&wb);
    return err;
}

/**
 * @brief Generate orthogonal Q matrix of QL factorization
 *
 * Generate the M-by-N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * @param[in,out]  A
 *     On entry, the elementary reflectors as returned by armas_qlfactor().
 *     stored below diagonal of the M by N matrix A.
 *     On exit, the orthogonal matrix Q
 *
 * @param[in]  tau
 *    Scalar coefficents of elementary reflectors
 *
 * @param[in]   K
 *     The number of elementary reflector whose product define the matrix Q
 *
 * @param[out] wb
 *    Workspace. If *wb.bytes* is zero then the size of required workspace
 *    is computed and returned immediately.
 *
 * @param[in,out] conf
 *     Blocking configuration
 *
 * @retval  0 Success
 * @retval <0 Failure, conf.error holds error code.
 *
 *  Last error codes returned
 *   - `ARMAS_EINVAL` A or tau is null pointer
 *   - `ARMAS_EWORK`  if no workspace or it is less than required for unblocked computation
 *
 *
 * Compatible with lapackd.ORGQL.
 * @ingroup lapack
 */
int armas_qlbuild_w(armas_dense_t * A,
                      const armas_dense_t * tau,
                      int K, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_dense_t T, Wrk;
    armas_env_t *env;
    size_t wsmin, wsz = 0;
    int lb;
    DTYPE *buf;

    if (!conf)
        conf = armas_conf_default();

    if (!A || !tau) {
        conf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    env = armas_getenv();
    if (wb && wb->bytes == 0) {
        if (env->lb > 0 && A->cols > env->lb)
            wb->bytes = (A->cols * env->lb) * sizeof(DTYPE);
        else
            wb->bytes = A->cols * sizeof(DTYPE);
        return 0;
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
        unblk_qlbuild(A, (armas_dense_t *) tau, &Wrk, A->rows-K,
                      A->cols-K, TRUE, conf);
    } else {
        // block reflector [lb, lb]; temporary space  [N(A)-lb, lb] matrix
        armas_make(&T, lb, lb, lb, buf);
        armas_make(&Wrk, A->cols-lb, lb, A->cols-lb, &buf[armas_size(&T)]);

        blk_qlbuild(A, (armas_dense_t *) tau, &T, &Wrk, K, lb, conf);
    }
    return 0;
}
#else
#warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
