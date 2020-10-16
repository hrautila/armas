
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_pgmres_w) && defined(armassp_pgmres)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_mvmult)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include <float.h>
#include <math.h>
#include "armas.h"
#include "dlpack.h"
#include "matrix.h"
#include "sparse.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"
#include "splocal.h"

/*
 * References:
 *  (1) Youssef Saad, Iterative Methods for Sparse Linear Systems, 2nd Edition
 *
 *  For more details in (1) section 9.3.2 algorithm 9.5.
 */

#ifndef __wbnil
#define __wbnil (armas_wbuf_t *)0
#endif

static
void hmult(armas_dense_t * y, armas_dense_t * H, int k, DTYPE val,
           armas_conf_t * cnf)
{
    armas_dense_t Hc, yc;
    armas_submatrix(&Hc, H, k, k, H->rows - k, 1);
    armas_subvector(&yc, y, k, H->rows - k);
    val += armas_get_unsafe(y, k, 0);
    armas_set_unsafe(y, k, 0, val);

    armas_housemult_w(&yc, __nil, &Hc, ARMAS_LEFT, __wbnil, cnf);
}

/*
 * GMRES with Householder orthogonalization (algorithm 6.10 in (1)) with
 * fused Hessenberg matrix reduction to triangular matrix.
 *
 * On entry w holds the residual r0 = b - A*x0
 *
 * \param [in] x
 *    Initial value for this iteration, nx1 matrix
 * \param [in] A  
 *    Matrix, nxn
 * \param [in] b
 *    Matrix, nx1
 * \param [out] P
 *    Unscaled Householder reflectors stored in diagonal and subdiagonal 
 *    entries of P. Intermediate triangularized Hessenberg matrix stored
 *    in strictly upper right triangular part of P. Matrix size nx(m+1).
 * \param [out] C, S
 *    Plane rotations for zeroing first subdiagonal of intermediate
 *    Hessenberg matrix (length m+1)
 * \param [out] y
 *    Intermediate result vector (length m+1)
 * \param [out] w
 *    On entry residual b - A*x. On exit values destroyed. (length n);
 * \param [out] *res
 *    Error residual; |beta*e_0 - Hy|
 */
static
int pgmres_hh_pfused(armas_dense_t * x,
                     const armas_sparse_t * A,
                     const armas_dense_t * b,
                     const armassp_precond_t * M,
                     armas_dense_t * P,
                     armas_dense_t * C,
                     armas_dense_t * S,
                     armas_dense_t * y,
                     armas_dense_t * w,
                     DTYPE * res, DTYPE stop, armas_conf_t * cnf)
{
    armas_dense_t PTL, PBL, PBR, P00, p01, p11, p21, P22, PL, PR, P0, p1, P2;
    armas_dense_t T, z, beta;
    DTYPE _beta, u, h0, c, s, r;
    int m = P->cols - 1;

    EMPTY(P00);

    // w = b - A*x0; z = r0
    armas_make(&beta, 1, 1, 1, &_beta);

    mat_partition_2x2(
        &PTL, __nil,
        &PBL, &PBR, /**/ P, 1, 1, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &PL, &PR, /**/ P, 1, ARMAS_PLEFT);

    armas_copy(&PL, w, cnf);
    armas_house(&PTL, &PBL, y, 0, cnf);
    int nrot = 0;
    int converged = 0;
    while (PBR.cols > 0 && converged == 0) {
        mat_repartition_2x2to3x3(
            &PTL, /**/
            &P00, &p01, __nil,
            __nil, &p11, __nil,
            __nil, &p21, &P22, /**/ P, 1, ARMAS_PBOTTOMRIGHT);
        mat_repartition_1x2to1x3(
            &PL, /**/ &P0, &p1, &P2, /**/ P, 1, ARMAS_PRIGHT);
        // ---------------------------------------------------------------------
        // w = v = P_1*..P_j*e_j = Q_j*e_j
        armas_scale(w, ZERO, cnf);
        armas_set_unsafe(w, P00.rows - 1, 0, ONE);
        armas_housemult_w(w, __nil, &P0, ARMAS_LEFT, __wbnil, cnf);
        // z = AM^-1*v; p1 == z
        M->precond(&p1, M, &p1, cnf);
        armassp_mvmult(ZERO, &p1, ONE, A, w, 0, cnf);
        // z = P_j*...P_1*Av
        armas_housemult_w(&p1, __nil, &P0, ARMAS_LEFT | ARMAS_TRANS, __wbnil,
                            cnf);

        // apply previouse rotations to p01
        armas_gvupdate(&p01, 0, C, S, nrot, ARMAS_LEFT | ARMAS_FORWARD);

        // unscaled Householder transformation to zero p21
        armas_house(&p11, &p21, &beta, 0, cnf);

        // compute rotation to zero new beta 
        h0 = armas_get_unsafe(&p01, p01.rows - 1, 0);
        armas_gvcompute(&c, &s, &r, h0, _beta);
        armas_set_unsafe(&p01, p01.rows - 1, 0, r);
        // clear y element from old value and rotate
        armas_set_unsafe(y, nrot + 1, 0, ZERO);
        armas_gvleft(y, c, s, nrot, nrot + 1, 0, 1);

        armas_set_unsafe(C, nrot, 0, c);
        armas_set_unsafe(S, nrot, 0, s);
        nrot++;
        converged = ABS(armas_get_unsafe(y, nrot, 0)) < stop;
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &PTL, __nil,
            &PBL, &PBR, /**/ &P00, &p11, &P22, /**/ P, ARMAS_PBOTTOMRIGHT);
        mat_continue_1x3to1x2(
            &PL, &PR, /**/ &P0, &p1, /**/ P, ARMAS_PRIGHT);
    }

    if (nrot < m) {
        m = nrot;
    }
    // residual of |beta*e0 - H*y| in y[m]
    *res = ABS(armas_get_unsafe(y, m, 0));

    // 2. solve for y; len(y) == cols(H) == m;
    armas_submatrix(&T, P, 0, 1, m, m);
    armas_make(&z, m, 1, m, armas_data(y));
    armas_mvsolve_trm(&z, ONE, &T, ARMAS_UPPER, cnf);

    // Compute:
    // z = 0; z = P_j*(y_j*e_j + z) for j in m ... 1

    // z = rightmost column of P
    armas_column(&z, P, P->cols - 1);
    armas_scale(&z, ZERO, cnf);

    for (int j = m - 1; j >= 0; j--) {
        u = armas_get_unsafe(y, j, 0);
        // compute z = P_j*(y_j*e_j + z) ; u = y_j*e_j
        hmult(&z, P, j, u, cnf);
    }
    // x = x + M^-1*z
    M->precond(&z, M, &z, cnf);
    armas_axpy(x, ONE, &z, cnf);
    return m;
}


#ifndef GMRES_M
#define GMRES_M 6
#endif

static inline
int GMRES_WSIZE(int n, int m)
{
    return (n + 3) * (m + 2) * sizeof(DTYPE);
}

static inline
int compute_m(int n, int w)
{
    if (w < n)
        return 0;
    int m = w / (n + 3) - 2;
    return m < 3 ? 0 : m;
}


static inline
int check_params(const armas_dense_t * x,
                 const armas_sparse_t * A, const armas_dense_t * b)
{
    if (!A || !x || !b)
        return 0;
    if (A->rows != A->cols)
        return 0;
    if (!armas_isvector(x))
        return 0;
    if (!armas_isvector(b))
        return 0;
    if (armas_size(x) != A->rows)
        return 0;
    if (armas_size(b) != A->cols)
        return 0;
    return 1;
}


/**
 * @brief Solve unsymmetric linear system A*x = b with right preconditioned GMRES algorithm.
 *
 * @param [in,out] x
 *    On entry, initial guess for solution. On exit solution the linear system.
 * @param [in] A
 *    Sparse coefficients of the linear system.
 * @param [in] b
 *    Dense vector
 * @param [in] M
 *    Preconditioner for unsymmetic linear system. See armassp_init_iluz for
 *    an ILU(0)  preconditioner.
 * @param [in,out] W
 *    Workspace buffer. If workspace size (W.bytes) is zero, then required
 *    workspace is calculated and returned in W.bytes.
 * @param [in,out] cf
 *    Configuration block - iteration parameters `gmres_m`, `maxiters`,
 *   `stop` and `smult`. Number of iterations  used is returned in `numiters`
 *    and error residual in `residual`.
 *
 *  Iteration stops when `maxiters` iterations is reached or when error residual
 *  \$ ||beta*e_1 - H_{m}y||_2 \$ is less than the stopping criteria. Stopping
 *  is criteria is non-zero absolute value defined in `stop` or value
 *  of `smult`*||b - Ax_0||_2.
 *
 *  For detailts see:
 *  Youssef Saad, *Iterative Methods for Sparse Linear Systems, 2nd Edition*, section 9.3
 *
 * @retval  0  Succress
 * @retval <0  Failure
 * @ingroup sparse
 */
int armassp_pgmres_w(armas_dense_t * x,
                       const armas_sparse_t * A,
                       const armas_dense_t * b,
                       const armassp_precond_t * M,
                       armas_wbuf_t * W, armas_conf_t * cf)
{
    int n, m, me, maxiter, niter;
    armas_dense_t P, w, C, S, y;
    DTYPE nrm_r0, nrm_r1, rstop, *t;

    if (!cf)
        cf = armas_conf_default();
    if (!A) {
        return -1;
    }

    n = A->cols;
    m = cf->gmres_m > 0 ? cf->gmres_m : GMRES_M;
    if (W->bytes == 0) {
        // get working size
        W->bytes = GMRES_WSIZE(n, m);
        return 0;
    }
    if (!check_params(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    // estimate of m
    me = compute_m(n, armas_wbytes(W) / sizeof(DTYPE));
    if (me == 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    maxiter = cf && cf->maxiter > 0 ? cf->maxiter : 5 * A->rows;

    // ------------------------------------------------------------------------
    size_t pos = armas_wpos(W);
    // matrix P; Householder vectors
    t = armas_wreserve(W, n * (m + 1), sizeof(DTYPE));
    armas_make(&P, n, m + 1, n, t);

    // Saved rotations
    t = armas_wreserve(W, m + 1, sizeof(DTYPE));
    armas_make(&C, m + 1, 1, m + 1, t);
    t = armas_wreserve(W, m + 1, sizeof(DTYPE));
    armas_make(&S, m + 1, 1, m + 1, t);

    t = armas_wreserve(W, m + 1, sizeof(DTYPE));
    armas_make(&y, m + 1, 1, m + 1, t);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_make(&w, n, 1, n, t);

    // w = b - A*x
    armas_copy(&w, b, cf);
    armassp_mvmult(ONE, &w, -ONE, A, x, 0, cf);
    nrm_r0 = armas_nrm2(&w, cf);

    rstop = cf->stop > ZERO ? cf->stop
        : (cf->smult > ZERO ? cf->smult * nrm_r0 : EPS * nrm_r0);

    // ------------------------------------------------------------------------
    nrm_r1 = ZERO;
    niter = 0;
    for (int j = 0; j < maxiter; j += m) {
        niter +=
            pgmres_hh_pfused(x, A, b, M, &P, &C, &S, &y, &w, &nrm_r1, rstop, cf);
        if (nrm_r1 < rstop) {
            break;
        }
        // compute residual; w = b - A*x
        armas_copy(&w, b, cf);
        armassp_mvmult(ONE, &w, -ONE, A, x, 0, cf);
    }

    armas_wsetpos(W, pos);

    cf->residual = nrm_r1;
    cf->numiters = niter;

    return 0;
}

/**
 * @brief Solve unsymmetric linear system A*x = b with right preconditioned GMRES algorithm
 * @see armassp_pgmres_w
 * @ingroup sparse
 */
int armassp_pgmres(armas_dense_t * x,       /*  */
                     const armas_sparse_t * A,
                     const armas_dense_t * b,
                     const armassp_precond_t * M, armas_conf_t * cf)
{
    armas_wbuf_t W = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!check_params(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }

    int m = cf->gmres_m > 0 ? cf->gmres_m : GMRES_M;

    int nb = GMRES_WSIZE(A->cols, m);
    if (!armas_walloc(&W, nb)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    int stat = armassp_pgmres_w(x, A, b, M, &W, cf);
    armas_wrelease(&W);
    return stat;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
