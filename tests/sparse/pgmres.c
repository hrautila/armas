
// Copyright by libARMAS authors. See AUTHORS file in this archive.

#include <stdio.h>
#include <unistd.h>
#include "testing.h"
#include "internal.h"
#include "partition.h"

#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-14
#endif


#ifndef __wbnil
#define __wbnil (armas_wbuf_t *)0
#endif

static void
__hmult(armas_dense_t *y, armas_dense_t *H, int k, DTYPE val, armas_conf_t *cnf)
{
    armas_dense_t Hc, yc;
    armas_submatrix(&Hc, H, k, k, H->rows-k, 1);
    armas_subvector(&yc, y, k, H->rows-k);
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
int __pgmres_hh_pfused(armas_dense_t *x,
                       const armas_sparse_t *A,
                       const armas_dense_t *b,
                       const armassp_precond_t *M,
                       armas_dense_t *P,
                       armas_dense_t *C,
                       armas_dense_t *S,
                       armas_dense_t *y,
                       armas_dense_t *w,
                       DTYPE *res,
                       DTYPE stop,
                       armas_conf_t *cnf)
{
    armas_dense_t PTL, PBL, PBR, P00, p01, p11, p21, P22, PL, PR, P0, p1, P2, beta;
    armas_dense_t T, z;
    DTYPE _beta, u, h0, c, s, r;
    int m = P->cols - 1;
    
    //EMPTY(P00);
    
    // w = b - A*x0; z = r0
    armas_make(&beta, 1, 1, 1, &_beta);
    
    __partition_2x2(&PTL, __nil,
                    &PBL,  &PBR,  /**/ P, 1, 1, ARMAS_PTOPLEFT);
    __partition_1x2(&PL,   &PR,   /**/ P, 1, ARMAS_PLEFT);
    
    armas_copy(&PL, w, cnf);
    armas_house(&PTL, &PBL, y, 0, cnf);
    int nrot = 0;
    int converged = 0;
    while (PBR.cols > 0 && converged == 0) {
        __repartition_2x2to3x3(&PTL,     /**/
                               &P00,   &p01, __nil,
                               __nil,  &p11, __nil,
                               __nil,  &p21,  &P22,          /**/ P, 1, ARMAS_PBOTTOMRIGHT);
        __repartition_1x2to1x3(&PL,      /**/ &P0, &p1, &P2, /**/ P, 1, ARMAS_PRIGHT);
        // -----------------------------------------------------------------------------
        // w = v = P_1*..P_j*e_j = Q_j*e_j
        armas_scale(w, __ZERO, cnf);
        armas_set_unsafe(w, P00.rows-1, 0, __ONE);
        armas_housemult_w(w, __nil, &P0, ARMAS_LEFT, __wbnil, cnf);
        // z = AM^-1v; p1 == z
        M->precond(w, M, w, cnf);
        armassp_mvmult(__ZERO, &p1, __ONE, A, w, 0, cnf);
        // z = P_j*...P_1*Av
        armas_housemult_w(&p1, __nil, &P0, ARMAS_LEFT|ARMAS_TRANS, __wbnil, cnf);

        // apply previouse rotations to p01
        armas_gvupdate(&p01, 0, C, S, nrot, ARMAS_LEFT|ARMAS_FORWARD);

        // unscaled Householder transformation to zero p21
        armas_house(&p11, &p21, &beta, 0, cnf);

        // compute rotation to zero new beta 
        h0 = armas_get_unsafe(&p01, p01.rows-1, 0);
        armas_gvcompute(&c, &s, &r, h0, _beta);
        armas_set_unsafe(&p01, p01.rows-1, 0, r);
        // clear y element from old value and rotate
        armas_set_unsafe(y, nrot+1, 0, __ZERO);
        armas_gvleft(y, c, s, nrot, nrot+1, 0, 1);

        armas_set_unsafe(C, nrot, 0, c);
        armas_set_unsafe(S, nrot, 0, s);
        nrot++;
        converged = __ABS(armas_get_unsafe(y, nrot, 0)) < stop;
        // -----------------------------------------------------------------------------
        __continue_3x3to2x2(&PTL, __nil,
                            &PBL,  &PBR, /**/ &P00, &p11, &P22, /**/ P, ARMAS_PBOTTOMRIGHT);
        __continue_1x3to1x2(&PL, &PR,    /**/ &P0,  &p1,        /**/ P, ARMAS_PRIGHT);
    }

    if (nrot < m) {
        m = nrot;
    }
    // residual of |beta*e0 - H*y| in y[m]
    *res = __ABS(armas_get_unsafe(y, m, 0));

    // 2. solve for y; len(y) == cols(H) == m;
    armas_submatrix(&T, P, 0, 1, m, m);
    armas_make(&z, m, 1, m, armas_data(y));
    armas_mvsolve_trm(&z, __ONE, &T, ARMAS_UPPER, cnf);

    // Compute:
    // z = 0; z = P_j*(y_j*e_j + z) for j in m ... 1

    // z = rightmost column of P
    armas_column(&z, P, P->cols-1);
    armas_scale(&z, __ZERO, cnf);
    
    for (int j = m-1; j >= 0; j--) {
        u = armas_get_unsafe(y, j, 0);
        // compute z = P_j*(y_j*e_j + z) ; u = y_j*e_j
        __hmult(&z, P, j, u, cnf);
    }
    // x = x + M^-1*z
    M->precond(&z, M, &z, cnf);
    armas_axpy(x, __ONE, &z, cnf);
    return m;
}


#ifndef __GMRES_M
#define __GMRES_M 6
#endif

static inline
int GMRES_WSIZE(int n, int m)
{
    return (n + 3)*(m + 2)*sizeof(DTYPE);
}

static inline
int __compute_m(int n, int w)
{
    if (w < n)
        return 0;
    int m = w / (n + 3) - 2;
    return m < 3 ? 0 : m;
}


static inline
int __check_params(const armas_dense_t *x,
                   const armas_sparse_t *A, const armas_dense_t *b)
{
    if (!A || !x || !b)       return 0;
    if (A->rows != A->cols) return 0;
    if (!armas_isvector(x)) return 0;
    if (!armas_isvector(b)) return 0;
    if (armas_size(x) != A->rows) return 0;
    if (armas_size(b) != A->cols) return 0;
    return 1;
}


/**
 * \brief Solve unsymmetric linear system A*x = b with GMRES algorithm
 */
int armassp_pgmres_w(armas_dense_t *x,
                       const armas_sparse_t *A,
                       const armas_dense_t *b,
                       const armassp_precond_t *M,
                       armas_wbuf_t *W,
                       armas_conf_t *cf)
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
    m = cf->gmres_m > 0 ? cf->gmres_m : __GMRES_M;
    if (W->bytes == 0) {
        // get working size
        W->bytes = GMRES_WSIZE(n, m);
        return 0;
    }    
    if (! __check_params(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    // estimate of m
    me = __compute_m(n, armas_wbytes(W)/sizeof(DTYPE));
    if (me == 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    maxiter = cf && cf->maxiter > 0 ? cf->maxiter : 5*A->rows;

    // ------------------------------------------------------------------------
    size_t pos = armas_wpos(W);
    // matrix P; Householder vectors
    t = armas_wreserve(W, n*(m+1), sizeof(DTYPE));
    armas_make(&P, n, m+1, n, t);
    
    // Saved rotations
    t = armas_wreserve(W, m+1, sizeof(DTYPE));
    armas_make(&C, m+1, 1, m+1, t);
    t = armas_wreserve(W, m+1, sizeof(DTYPE));
    armas_make(&S, m+1, 1, m+1, t);

    t = armas_wreserve(W, m+1, sizeof(DTYPE));
    armas_make(&y, m+1, 1, m+1, t);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_make(&w, n, 1, n, t);
    
    // w = b - A*x
    armas_copy(&w, b, cf);
    armassp_mvmult(__ONE, &w, -__ONE, A, x, 0, cf);
    nrm_r0 = armas_nrm2(&w, cf);

    rstop = cf->stop > __ZERO ? cf->stop
        : (cf->smult > 0.0D ? cf->smult*nrm_r0 : __EPSILON*nrm_r0);
    
    // ------------------------------------------------------------------------
    nrm_r1 = __ZERO;
    niter = 0;
    for (int j = 0; j < maxiter; j += m) {
        niter += __pgmres_hh_pfused(x, A, b, M, &P, &C, &S, &y, &w, &nrm_r1, rstop, cf);
        printf("%02d |r0|: %9.2e, |r1|: %9.2e [%9.4e] rstop: %9.2e\n", j, nrm_r0, nrm_r1, nrm_r1/nrm_r0, rstop);
        //printf("%3d:  stop %e, nrm_r1 %e\n", j, rstop, nrm_r1);
        if (nrm_r1 < rstop) {
            break;
        }
        // compute residual; w = b - A*x
        armas_copy(&w, b, cf);
        armassp_mvmult(__ONE, &w, -__ONE, A, x, 0, cf);
    }

    armas_wsetpos(W, pos);

    cf->residual = nrm_r1;
    cf->numiters = niter;

    return 0;
}



// compute: ||x - A^-1*(A*x)||
int test_gen(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind, int maxiter, int m)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y, z, Cd, t;
    armassp_d_precond_t M;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    armas_wbuf_t wb;
    int stat = 0;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";

    int n = A->cols;
    int p = (n+1)*(m+1) + m*(m+1);
    int p1 = (n+3)*(m+2);

    armas_walloc(&wb, sizeof(double)*p1*2);

    C  = armassp_d_convert(A, kind);
    Cu = armassp_d_mkcopy(C);
    
    // x and y 
    armas_d_init(&x, C->rows, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_init(&z, C->rows, 1);
    armas_d_set_values(&x, one, 0);
    
    armas_d_init(&Cd, C->rows, C->cols);
    
    armassp_d_todense(&Cd, C, &cf);
    armassp_d_mvmult(0.0, &y,  1.0, C, &x, flags, &cf);

    //fprintf(stderr, "A\n");  armas_d_printf(stderr, "%.2e", &Cd);  
    //fprintf(stderr, "A*x\n"); armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&t, &y));
  
    armassp_d_init_iluz(&M, Cu);
    
    cf.maxiter = maxiter;
    cf.gmres_m = m;
    armassp_pgmres_w(&z, C, &y, &M, &wb, &cf);

    if (verbose > 1) {
        armas_d_dense_t tmp;
        fprintf(stderr, "y = A*x:\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &y));
        fprintf(stderr, "z = A^-1*y:\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
    }

    relerr = rel_error(&nrm, &z, &x, ARMAS_NORM_INF, 0, &cf);

    fprintf(stderr, "%s (trans=%s) ||x - A^-1*(A*x)||: %e\n", st, sf, relerr);
    if (verbose) {
        armas_d_dense_t tmp;
        fprintf(stderr, "x - A^-1*(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
    }

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armassp_d_free(C);
    armas_wrelease(&wb);
    return stat;
}


int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = (char *)0;
    FILE *fp;
    int tc;
    int gmres_m = 6;
    int maxiter = 36;
    armas_d_sparse_t *A;

    while ((opt = getopt(argc, argv, "vf:n:m:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'f':
            path = optarg;
            break;
        case 'm':
            gmres_m = atoi(optarg);
            break;
        case 'n':
            maxiter = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: tstmul [-v -f path] \n");
            exit(1);
        }
    }
    
    A = (armas_d_sparse_t *)0;
    if (path) {
        if (verbose > 0)
            fprintf(stderr, "opening '%s'...\n", path);

        if (! (fp = fopen(path, "r"))) {
            perror(path);
            exit(1);
        }
        A = armassp_d_mmload(&tc, fp);
        if (!A) {
            fprintf(stderr, "reading of '%s' failed\n", path);
            exit(1);      
        }
    }
    test_gen(A, 0, verbose, ARMASSP_CSR, maxiter, gmres_m);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
