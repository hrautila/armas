
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_gmres_w) && defined(armassp_gmres)
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
 * For more details in (1) section 6.5.
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
int gmres_hh_fused(armas_dense_t * x,
                   const armas_sparse_t * A,
                   const armas_dense_t * b,
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
        // z = Av; p1 == z
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
    armas_axpy(x, ONE, &z, cnf);
    return m;
}

/* 
 * Main loop for GMRES. Workspace needed is approxmately (n+3)*(m+2) elements.
 */
static
int gmres_hh_loop_fused(armas_dense_t * x,
                        const armas_sparse_t * A,
                        const armas_dense_t * b,
                        int maxiter,
                        int m,
                        DTYPE * stopping, armas_wbuf_t * W, armas_conf_t * cf)
{
    armas_dense_t P, w, C, S, y;
    DTYPE nrm_r0, nrm_r1, rstop;
    int n = A->cols;

    size_t pos = armas_wpos(W);
    // matrix P; Householder vectors
    DTYPE *t = armas_wreserve(W, n * (m + 1), sizeof(DTYPE));
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

    rstop = stopping && *stopping != ZERO ?
        *stopping : EPS * nrm_r0 * A->cols;

    int niter = 0;
    for (int j = 0; j < maxiter; j += m) {
        niter +=
            gmres_hh_fused(x, A, b, &P, &C, &S, &y, &w, &nrm_r1, rstop, cf);
        if (nrm_r1 < rstop) {
            break;
        }
        // compute residual; w = b - A*x
        armas_copy(&w, b, cf);
        armassp_mvmult(ONE, &w, -ONE, A, x, 0, cf);
    }

    armas_wsetpos(W, pos);
    return niter;
}

#ifdef ARMASSP_INCLUDE_GMRES_MGS_VERSION
/*
 * GMRES with MGS as algorithm 6.9 in (1)
 */
static
int gmres_mgs(armas_dense_t * x,
              const armas_sparse_t * A,
              const armas_dense_t * b,
              armas_dense_t * V,
              armas_dense_t * H,
              armas_dense_t * w, DTYPE * res, armas_conf_t * cnf)
{
    int m = V->cols;
    DTYPE beta, h_ij, h_jp1;
    armas_dense_t y, v, h, vi, T;

    //__pr(x, "x0"); __pr(b, "b ");

    // w = b - A*x0; v1 = r0/|r0|_2
    armas_column(&v, V, 0);
    armas_copy(&v, w, cnf);   // w = b - A*x0
    beta = armas_nrm2(&v, cnf);
    armas_scale(&v, 1.0 / beta, cnf);

    for (int j = 0; j < m; j++) {
        armassp_mvmult(ZERO, w, ONE, A, &v, 0, cnf);

        armas_column(&h, H, j);
        for (int i = 0; i <= j; i++) {
            armas_column(&vi, V, i);
            h_ij = armas_dot(w, &vi, cnf);
            armas_set_at_unsafe(&h, i, h_ij);
            armas_axpy(w, -h_ij, &vi, cnf);
        }
        h_jp1 = armas_nrm2(w, cnf);
        armas_set_at(&h, j + 1, h_jp1);
        if (h_jp1 == 0.0) {
            m = j + 1;
            break;
        }
        if (j < m - 1) {
            armas_column(&v, V, j + 1);
            armas_mcopy(&v, w);
            armas_scale(&v, 1.0 / h_jp1, cnf);
        }
    }
    // y = beta*e0
    armas_make(&y, m + 1, 1, m + 1, armas_data(w));
    armas_scale(&y, ZERO, cnf);
    armas_set_unsafe(&y, 0, 0, beta);

    // compute minimizer of |beta*e0 - H*y|
    // 1. transform H to upper triangular by Givens rotations
    for (int j = 0; j < m; j++) {
        DTYPE c, s, r, h0, h1;
        h0 = armas_get_unsafe(H, j, j);
        h1 = armas_get_unsafe(H, j + 1, j);
        armas_gvcompute(&c, &s, &r, h0, h1);
        armas_set_unsafe(H, j, j, r);
        armas_set_unsafe(H, j + 1, j, ZERO);
        // apply rotation to rest of the rows j,j+1
        armas_gvleft(H, c, s, j, j + 1, j + 1, m - j - 1);
        armas_gvleft(&y, c, s, j, j + 1, 0, 1);
    }

    *res = armas_get_unsafe(&y, m, 0);

    // 2. solve for y
    armas_make(&y, m, 1, m, armas_data(w));
    armas_submatrix(&T, H, 0, 0, m, m);
    armas_mvsolve_trm(&y, ONE, &T, ARMAS_UPPER, cnf);
    // compute x = x0 + V*y
    armas_mvmult(ONE, x, ONE, V, &y, 0, cnf);
    return m;
}

static
int gmres_loop(armas_dense_t * x,
               const armas_sparse_t * A,
               const armas_dense_t * b,
               int maxiter, int m, armas_wbuf_t * W, armas_conf_t * cf)
{
    armas_dense_t V, H, w;
    DTYPE nrm_r0, nrm_r1, rstop;
    int n = A->cols;

    // matrix V  (size: n*m)
    DTYPE *t = armas_wreserve(W, n * m, sizeof(DTYPE));
    armas_make(&V, n, m, n, t);

    // Hessenberg matrix (size: (m+1)*m)
    t = armas_wreserve(W, (m + 1) * m, sizeof(DTYPE));
    armas_make(&H, m + 1, m, m + 1, t);

    // intermediate result vector (size: n)
    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_make(&w, n, 1, n, t);

    // 
    // w = b - A*x
    armas_copy(&w, b, cf);
    armassp_mvmult(ONE, &w, -ONE, A, x, 0, cf);
    nrm_r0 = armas_nrm2(&w, cf);

    rstop = SQRT(EPSILON) * nrm_r0;     // * A->cols;

    int stat = 0;
    for (int j = 0; j < maxiter; j += m) {
        stat += gmres_mgs(x, A, b, &V, &H, &w, &nrm_r1, cf);

        // if (nrm_r1 < rstop)
        if (nrm_r1 / nrm_r0 < EPSILON) {
            break;
        }
        // compute residual; w = b - A*x
        armas_copy(&w, b, cf);
        armassp_mvmult(ONE, &w, -ONE, A, x, 0, cf);
    }

    armas_wreset(W);
    return stat;
}

static inline
int compute_mgs_m(int n, int w)
{
    // compute an estimate of gmres parameter m for work space size w elements;
    // space: n*m + (m+1)*m + n ==> m^2 + (n+1)*m + n < w ==> m^2 + (n+1)*m + n - w = 0
    // m = max(-(n+1) +/- sqrt( (n+1)^2 - 4*(n-w) )) / 2
    double r = sqrt((n + 1) * (n + 1) - 4 * (n - w));
    if (r < (double) (n + 1))
        return 0;
    return ((int) floor(r - (double) (n + 1))) / 2;
}

static inline
int GMRES_WSIZE_MGS(int n, int m)
{
    return (n * m + (m + 1) * m + n) * sizeof(DTYPE);
}
#endif  /* ARMASS_INCLUDE_GMRES_MGS_VERSION */

#ifndef GMRES_M
#define GMRES_M 6
#endif

static inline
int GMRES_WSIZE(int n, int m)
{
    //return (n*m + (m+1)*m + n)*sizeof(DTYPE);
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
int check_params(armas_dense_t * x,
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
 * @brief Solve unsymmetric linear system A*x = b with GMRES algorithm
 *
 * @param[out] x
 *   On exit solution to the system.
 * @param[in]  A
 *   Sparse matrix.
 * @param[in]  b
 *   Dense vector.
 * @param[in,out] wb
 *   Workspace. If *wb.bytes* is zero then size of required workspace is calculated
 *   and returned immediately.
 * @param[in,out] cf
 *   Configuration block. On exit *cf.numiters* holds number of iterations and
 *   *cf.residual* the final error residual.
 *
 * For details see: Yousef Saad, *Iterative Methods for Sparse Linear System*, 2nd Edition
 * sections 6.5 and algorithm 6.9.
 *
 * @retval  0  Success
 * @retval <0 Failure
 * @ingroup sparse
 */
int armassp_gmres_w(armas_dense_t * x,
                      const armas_sparse_t * A,
                      const armas_dense_t * b,
                      armas_wbuf_t * wb, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int n = A->rows;
    int m = cf->gmres_m > 0 ? cf->gmres_m : GMRES_M;

    if (wb && wb->bytes == 0) {
        // get working size
        wb->bytes = GMRES_WSIZE(n, m);
        return 0;
    }
    if (!check_params(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    // estimate of m
    int me = compute_m(n, armas_wbytes(wb) / sizeof(DTYPE));
    if (me == 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    int maxiter = cf->maxiter > 0 ? cf->maxiter : 5 * A->rows;
    DTYPE res = ZERO;
    int niter = gmres_hh_loop_fused(x, A, b, maxiter, me, &res, wb, cf);
    cf->residual = res;
    cf->numiters = niter;
    return 0;
}

/**
 * @brief Solve unsymmetric linear system A*x = b with GMRES algorithm
 * @see armassp_gmres_w
 * @ingroup sparse
 */
int armassp_gmres(armas_dense_t * x,
                    const armas_sparse_t * A,
                    const armas_dense_t * b, armas_conf_t * cf)
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
    int stat = armassp_gmres_w(x, A, b, &W, cf);
    armas_wrelease(&W);
    return stat;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
