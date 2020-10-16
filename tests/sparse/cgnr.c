
// Copyright by libARMAS authors. See AUTHORS file in this archive.

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-14
#endif


#ifndef __wbnil
#define __wbnil (armas_wbuf_t *)0
#endif

/* 
 * r0 = b - A*x0; z0 = A^T*r0; p0 = z0
 * for i ... until convergense
 *   w_j     = Ap_j
 *   alpha_j = (z_j, z_j)/(w_j, w_j)
 *   x_jp1   = x_j + alpha_j*p_j
 *   r_jp1   = r_j - alpha_j*w_j
 *   z_jp1   = A^T*r_jp1
 *   beta_j   = (z_jp1, z_jp1)/(z_j, z_j)
 *   p_jp1    = z_jp1  + beta_j*p_j
 * endfor
 *
 *  A     = [m,n] m > n
 *  x,z,p = [n,1]
 *  b,r,w = [m,1]
 */

static
int __x_cgnr(armas_dense_t *x,
             const armas_sparse_t *A,
             armas_dense_t *b,
             int maxiter,
             DTYPE rstop,
             DTYPE *res,
             armas_wbuf_t *W,
             armas_conf_t *cf)
{
    armas_dense_t p, Ap, r, z;
    int m = A->rows;
    int n = A->cols;
    DTYPE dot_z, dot_p, alpha, beta, dot_z1;

    // r0 = b - A*x
    DTYPE *t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&r, m, 1, m, t);
    armas_mcopy(&r, b);
    armassp_mvmult(__ONE, &r, -__ONE, A, x, 0, cf);

    if (rstop == __ZERO)
        rstop = __EPSILON * armas_nrm2(&r, cf);

    // z0 = A^T*r0
    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_make(&z, n, 1, n, t);
    armassp_mvmult(__ZERO, &z, __ONE, A, &r, ARMAS_TRANS, cf);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_make(&p, n, 1, n, t);
    armas_copy(&p, &z, cf);

    // w = Ap
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&Ap, m, 1, m, t);
    
    if (maxiter == 0)
        maxiter = 4*m;
    
    int niter = 0;
    for (int i = 0; i < maxiter; i++, niter++) {
        // Ap = A*p
        armassp_mvmult(__ZERO, &Ap, __ONE, A, &p, 0, cf);
        dot_z = armas_dot(&z, &z, cf);
        dot_p = armas_dot(&Ap, &Ap, cf);
        alpha = dot_z / dot_p;

        // x = x + alpha*p
        armas_axpy(x, alpha, &p, cf);
        // r = r - alpha*w = r - alpha*Ap
        armas_axpy(&r, -alpha, &Ap, cf);
        // z = A^T*r
        armassp_mvmult(__ZERO, &z, __ONE, A, &r, ARMAS_TRANS, cf);

        dot_z1 = armas_dot(&z, &z, cf);
        printf("%3d: stop %e, dot_z %e, dot_z1 %e, dot_p %e\n", i, rstop, dot_z, dot_z1, dot_p);
        if (__SQRT(dot_z1) < rstop) {
            if (res)
                *res = __SQRT(dot_z1);
            break;
        }
        beta = dot_z1 / dot_z;
        // p = beta*p + z;
        armas_scale_plus(beta, &p, __ONE, &z, 0, cf);
    }
    return niter;
}

// compute: ||x - A^-1*(A*x)||
int test_sym(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind, int maxiter)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y, z, Cd;
    armas_d_dense_t tmp;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    DTYPE rstop, res;
    armas_wbuf_t wb;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";
    char *uplo = (flags & ARMAS_UPPER) != 0 ? "U" : "L";
    int p = 2*A->cols + 2*A->rows;

    armas_walloc(&wb, sizeof(double)*p*2);
    
    C = armassp_d_convert(A, kind);
    printf("A = [%d,%d]\n", C->rows, C->cols);
    // x and y 
    armas_d_init(&x, C->cols, 1);
    armas_d_init(&z, C->cols, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_set_values(&x, one, 0);
    
    armas_d_init(&Cd, C->rows, C->cols);
    
    if ((flags & ARMAS_UPPER) != 0) {
        Cu = armassp_d_transpose(C);
        armassp_d_free(C);
        C = Cu;
    }
    armassp_d_todense(&Cd, C, &cf);
    armassp_d_mvmult(0.0, &y,  1.0, C, &x, 0, &cf);
    printf("y:\n"); armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &y));

    rstop = __ZERO;
    res  = __ZERO;
    __x_cgnr(&z, C, &y, maxiter, rstop, &res, &wb, &cf);

    if (verbose > 1) {
        armas_d_dense_t tmp;
        fprintf(stderr, "y = A*x:\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &y));
        fprintf(stderr, "z = A^-1*y:\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
    }

    relerr = rel_error(&nrm, &z, &x, ARMAS_NORM_INF, 0, &cf);

    fprintf(stderr, "%s (uplo=%s) ||x - A^-1*(A*x)||: %e\n", st, uplo, relerr);
    if (verbose) {
        fprintf(stderr, "x - A^-1*(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
    }

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armassp_d_free(C);
    return stat;
}


int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = "test1.mtx";
    FILE *fp;
    int tc;
    armas_d_sparse_t *A;
    int gmres_m = 6;
    int maxiter = 36;

    while ((opt = getopt(argc, argv, "vf:m:n:")) != -1) {
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
    
    if (verbose > 0)
        fprintf(stderr, "opening '%s'...\n", path);

    if (! (fp = fopen(path, "r"))) {
        perror(path);
        exit(1);
    }
    A = armassp_d_mmload(&tc, fp);


    test_sym(A, 0, verbose, ARMASSP_CSR, maxiter);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSR);
    //test_sym(A, ARMAS_LOWER, verbose, ARMASSP_CSC);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSC);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
