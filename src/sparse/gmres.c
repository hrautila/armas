
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_gmres_w) && defined(armassp_x_gmres)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_mvmult) 
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <float.h>
#include <math.h>
#include <armas/armas.h>
#include "dlpack.h"
#include "matrix.h"
#include "sparse.h"

#include "splocal.h"

/*
   r0 = b - A*x0
   beta = ||r0||_2
   v0   = r0/beta
   for j = 0,, m-1:
      w_j = A*v_j
      for i = 0 ... j:
        h_ij = (w_j, v_i)
        w_j = w_j - h_ij*v_i
      endfor
      h_jp1,j = ||w_j||_2
      if h_jp1,j == 0:
         m = j
         break
      endif
      v_jp1 = w_j/h_jp1,j
    endfor
    define (m+1)xm Hessenberg H_m = {h_ij}
    compute y_m = min ||beta*e1 - H_m*y||_2
    x_m = V_m*y_m
 */


#if 0
static void __pr(const armas_d_dense_t *x, const char *s)
{
    armas_d_dense_t t;
    printf("%s :", s);
    armas_d_printf(stdout, "%9.2e", armas_d_col_as_row(&t, (armas_d_dense_t *)x));
}

static void __pm(const armas_d_dense_t *m, const char *s)
{
    printf("%s :\n", s);
    armas_d_printf(stdout, "%9.2e", m);
}
#endif


static
int __gmres(armas_x_dense_t *x, const armas_x_sparse_t *A, const armas_x_dense_t *b,
            armas_x_dense_t *V, armas_x_dense_t *H, armas_x_dense_t *w, armas_conf_t *cnf)
{
    int m = V->cols;
    DTYPE beta, h_ij, h_jp1;
    armas_x_dense_t y, v, h, vi, T;

    //__pr(x, "x0"); __pr(b, "b ");

    // w = b - A*x0; v1 = r0/|r0|_2
    armas_x_column(&v, V, 0);
    armas_x_copy(&v, w, cnf);   // w = b - A*x0
    beta = armas_x_nrm2(&v, cnf);
    armas_x_scale(&v, 1.0/beta, cnf);
    
    for (int j = 0; j < m; j++) {
        armassp_x_mvmult(__ZERO, w, __ONE, A, &v, 0, cnf);

        armas_x_column(&h, H, j);
        for (int i = 0; i <= j; i++) {
            armas_x_column(&vi, V, i);
            h_ij = armas_x_dot(w, &vi, cnf);
            armas_x_set_at_unsafe(&h, i, h_ij);
            armas_x_axpy(w, -h_ij, &vi, cnf);
        }
        h_jp1 = armas_x_nrm2(w, cnf);
        armas_x_set_at(&h, j+1, h_jp1);
        if (h_jp1 == 0.0) {
            m = j+1;
            break;
        }
        if (j < m-1) {
            armas_x_column(&v, V, j+1);
            armas_x_mcopy(&v, w);
            armas_x_scale(&v, 1.0/h_jp1, cnf);
        }
    }
    // y = beta*e0
    armas_x_make(&y, m, 1, m, armas_x_data(w));
    armas_x_scale(&y, __ZERO, cnf);
    armas_x_set_unsafe(&y, 0, 0, beta);

    // compute minimizer of |beta*e0 - H*y|
    // 1. transform H to upper triangular by Givens rotations
    for (int j = 0; j < m; j++) {
        DTYPE c, s, r, h0, h1;
        h0 = armas_x_get_unsafe(H, j, j);
        h1 = armas_x_get_unsafe(H, j+1, j);
        armas_x_gvcompute(&c, &s, &r, h0, h1);
        armas_x_set_unsafe(H, j, j, r);
        armas_x_set_unsafe(H, j+1, j, __ZERO);
        // apply rotation to rest of the rows j,j+1
        armas_x_gvleft(H, c, s, j, j+1, j+1, m-j-1);
        armas_x_gvleft(&y, c, s, j, j+1, 0, 1);
    }

    // 2. solve for y
    armas_x_submatrix(&T, H, 0, 0, m, m);
    armas_x_mvsolve_trm(&y, __ONE, &T, ARMAS_UPPER, cnf);
    // compute x = x0 + V*y
    armas_x_mvmult(__ONE, x, __ONE, V, &y, 0, cnf);
    return 0;
}

static 
int __gmres_loop(armas_x_dense_t *x,
                 const armas_x_sparse_t *A, const armas_x_dense_t *b,
                 int maxiter, int m,
                 armas_wbuf_t *W, armas_conf_t *cf)
{
    armas_x_dense_t V, H, w;
    DTYPE nrm_r0, nrm_r1, rstop;
    int n = A->cols;
    
    // matrix V
    DTYPE *t = armas_wreserve(W, n*m, sizeof(DTYPE));
    armas_x_make(&V, n, m, n, t);
    
    // Hessenberg matrix
    t = armas_wreserve(W, (m+1)*m, sizeof(DTYPE));
    armas_x_make(&H, m+1, m, m+1, t);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&w, n, 1, n, t);
    
    // w = b - A*x
    armas_x_copy(&w, b, cf);
    armassp_x_mvmult(__ONE, &w, -__ONE, A, x, 0, cf);
    nrm_r0 = armas_x_nrm2(&w, cf);

    rstop = __SQRT(__EPSILON) * nrm_r0 * A->cols;
    
    int stat = -1;
    for (int j = 0; j < maxiter; j += m) {
        __gmres(x, A, b, &V, &H, &w, cf);
        // compute residual
        armas_x_copy(&w, b, cf);
        armassp_x_mvmult(__ONE, &w, -__ONE, A, x, 0, cf);
        nrm_r1 = armas_x_nrm2(&w, cf);

        //printf("%02d |r0|: %9.2e, |r1|: %9.2e [%9.4e] rstop: %9.2e\n", j, nrm_r0, nrm_r1, nrm_r1/nrm_r0, rstop);
        //nrm_r0 = nrm_r1;
        if (nrm_r1 < rstop) {
            stat = 0;
            break;
        }
    }

    armas_wreset(W);
    return stat;
}

#ifndef __GMRES_M
#define __GMRES_M 6
#endif

static inline
int GMRES_WSIZE(int n, int m)
{
    return (n*m + (m+1)*m + n)*sizeof(DTYPE);
}

// compute an estimate of gmres parameter m for work space size w elements;
// space: n*m + (m+1)*m + n ==> m^2 + (n+1)*m + n < w ==> m^2 + (n+1)*m + n - w = 0
// m = max(-(n+1) +/- sqrt( (n+1)^2 - 4*(n-w) )) / 2
static inline
int __compute_m(int n, int w)
{
    if (w < n)
        return 0;
    double r = sqrt((n+1)*(n+1) - 4*(n - w));
    if (r < (double)(n+1))
        return 0;
    return ((int)floor(r - (double)(n+1)))/2;
}

static inline
int __check_params(armas_x_dense_t *x,
                   const armas_x_sparse_t *A, const armas_x_dense_t *b)
{
    return !x && !A && !b &&
        armas_x_isvector(x) &&
        armas_x_isvector(b) &&
        armas_x_size(x) == A->rows &&
        armas_x_size(b) == A->cols;
}


/**
 * \brief Solve unsymmetric linear system A*x = b with GMRES algorithm
 */
int armassp_x_gmres_w(armas_x_dense_t *x,
                      const armas_x_sparse_t *A,
                      const armas_x_dense_t *b,
                      int maxiter,
                      armas_wbuf_t *W,
                      armas_conf_t *cf)
{
    int n = A->rows;
    
    if (W->bytes == 0) {
        // get working size
        W->bytes = GMRES_WSIZE(n, __GMRES_M);
        return 0;
    }
    if (!cf)
        cf = armas_conf_default();
    
    if (! __check_params(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    // estimate of m
    int m = __compute_m(n, armas_wbytes(W)/sizeof(DTYPE));
    if (m == 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    if (maxiter == 0)
        maxiter = 5*A->rows;
    
    return __gmres_loop(x, A, b, maxiter, m, W, cf);
}

/**
 * \brief Solve unsymmetric linear system A*x = b with GMRES algorithm
 */
int armassp_x_gmres(armas_x_dense_t *x, /*  */
                    const armas_x_sparse_t *A,
                    const armas_x_dense_t *b,
                    int maxiter,
                    int m,
                    armas_conf_t *cf)
{
    armas_wbuf_t W = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!__check_params(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }

    if (m == 0)
        m = __GMRES_M;
    int nb = GMRES_WSIZE(A->cols, m);
    if (!armas_walloc(&W, nb)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    if (maxiter == 0)
        maxiter = 5*A->rows;
    
    int stat = __gmres_loop(x, A, b, maxiter, m, &W, cf);
    armas_wrelease(&W);
    return stat;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
