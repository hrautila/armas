
// Copyright by libARMAS authors. See AUTHORS file in this archive.

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

static
int __x_cgrad(armas_dense_t *x,
              const armas_sparse_t *A,
              armas_dense_t *b,
              int flags,
              int maxiter,
              int recalc,
              DTYPE rstop,
              DTYPE rmult,
              DTYPE *res,
              armas_wbuf_t *W,
              armas_conf_t *cf)
{
    armas_dense_t p, Ap, r;
    int m = A->rows;
    DTYPE dot_r, dot_p, alpha, beta, dot_r1;

    // r0 = b - A*x
    DTYPE *t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&r, m, 1, m, t);
    armas_mcopy(&r, b);
    armassp_mvmult_sym(__ONE, &r, -__ONE, A, x, flags, cf);

    if (rstop == __ZERO)
        rstop = (rmult == __ZERO ? __EPSILON : rmult) * armas_dot(&r, &r, cf);

    // p0 = r0
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&p, m, 1, m, t);
    armas_mcopy(&p, &r);

    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&Ap, m, 1, m, t);
    
    if (maxiter == 0)
        maxiter = 4*m;
    
    int niter;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_mvmult_sym(__ZERO, &Ap, __ONE, A, &p, flags, cf);
        dot_r = armas_dot(&r, &r, cf);
        dot_p = armas_dot(&Ap, &p, cf);
        alpha = dot_r / dot_p;

        // x = x + alpha*p
        armas_axpy(x, alpha, &p, cf);
        if (recalc > 0 && niter % recalc == 0) {
            armas_mcopy(&r, b);
            armassp_mvmult_sym(__ONE, &r, -__ONE, A, x, flags, cf);
        } else {
            // r = r - alpha*Ap
            armas_axpy(&r, -alpha, &Ap, cf);
        }

        dot_r1 = armas_dot(&r, &r, cf);
        printf("%3d: stop %e, dot_r %e, dot_r1 %e, dot_p %e\n", niter, rstop, dot_r, dot_r1, dot_p);
        if (dot_r1 < rstop) {
            if (res)
                *res = __SQRT(dot_r1);
            break;
        }
        beta = dot_r1 / dot_r;
        // p = beta*p + r;
        armas_scale_plus(beta, &p, __ONE, &r, 0, cf);
    }
    printf("%3d: stop %e, dot_r %e, dot_r1 %e, dot_p %e\n", niter, rstop, dot_r, dot_r1, dot_p);
    return niter;
}

// compute: ||x - A^-1*(A*x)||
int test_sym(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind,
             int maxiter, int recalc, double stop, double rmult)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y, z, Cd;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";
    char *uplo = (flags & ARMAS_UPPER) != 0 ? "U" : "L";

    DTYPE rstop, res;
    armas_wbuf_t wb;

    int p = 3*A->cols;

    armas_walloc(&wb, sizeof(double)*p*2);
    
    C = armassp_d_convert(A, kind);
    // x and y 
    armas_d_init(&x, C->rows, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_init(&z, C->rows, 1);
    armas_d_set_values(&x, one, 0);
    
    armas_d_init(&Cd, C->rows, C->cols);
    
    if ((flags & ARMAS_UPPER) != 0) {
        Cu = armassp_d_transpose(C);
        armassp_d_free(C);
        C = Cu;
    }
    armassp_d_todense(&Cd, C, &cf);
    armassp_d_mvmult_sym(0.0, &y,  1.0, C, &x, flags, &cf);

    rstop = stop;
    res = __ZERO;
    if (maxiter == 0)
        maxiter = 4*C->rows;
    
    __x_cgrad(&z, C, &y, flags, maxiter, recalc, rstop, rmult, &res, &wb, &cf);

    //if (armassp_d_cgrad(&z, C, &y, flags, &cf) < 0) {
    //fprintf(stderr, "cgrad error: %d\n", cf.error);
    //}
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
        armas_d_dense_t tmp;
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
    int tc, maxiter = 0;
    double stop = 0.0;
    double rmult = 0.0;
    int recalc = 0;
    armas_d_sparse_t *A;

    while ((opt = getopt(argc, argv, "vf:n:R:M:r:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'f':
            path = optarg;
            break;
        case 'n':
            maxiter = atoi(optarg);
            break;
        case 'r':
            recalc = atoi(optarg);
            break;
        case 'R':
            stop = strtod(optarg, (char **)0);
            break;
        case 'M':
            rmult = strtod(optarg, (char **)0);
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


    test_sym(A, ARMAS_LOWER, verbose, ARMASSP_CSR, maxiter, recalc, stop, rmult);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSR);
    //test_sym(A, ARMAS_LOWER, verbose, ARMASSP_CSC);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSC);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
