
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

static inline
void __pr_vec(const char *s, armas_x_dense_t *x, int n)
{
    armas_x_dense_t tx, x0;
    armas_x_submatrix(&x0, x, 0, 0, x->rows < n ? x->rows : n, 1);
    printf("%s: ", s);
    armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&tx, &x0));  
}

static
int __x_pcgrad(armas_x_dense_t *x,
               const armas_x_sparse_t *A,
               armas_x_dense_t *b,
               armassp_x_precond_t *M,
               int flags,
               int maxiter,
               DTYPE tolmult,
               DTYPE rstop,
               DTYPE *res,
               armas_wbuf_t *W,
               armas_conf_t *cf)
{
    armas_x_dense_t p, Ap, r, z;
    armas_x_dense_t ztmp, tmp;
    int m = A->rows;
    DTYPE dot_r, dot_p, alpha, beta, dot_rz1, dot_rz;

    // r0 = b - A*x
    DTYPE *t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&r, m, 1, m, t);
    armas_x_mcopy(&r, b);
    armassp_x_mvmult_sym(__ONE, &r, -__ONE, A, x, flags, cf);

    __pr_vec(" r0", &r, 10);

    if (rstop == __ZERO) 
        rstop = (tolmult == __ZERO ? __EPSILON*__EPSILON : tolmult*tolmult ) *armas_x_dot(&r, &r, cf);

    // z0 = M^-1*r0
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&z, m, 1, m, t);
    //armas_x_mcopy(&z, &r);
    M->precond(&z, M, &r, cf);
    
    __pr_vec(" z0", &z, 10);

    // p0 = z0
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&p, m, 1, m, t);
    armas_x_mcopy(&p, &z);

    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&Ap, m, 1, m, t);
    
    if (maxiter == 0)
        maxiter = m;
    
    int niter;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_x_mvmult_sym(__ZERO, &Ap, __ONE, A, &p, flags, cf);
        dot_p  = armas_x_dot(&Ap, &p, cf);
        dot_rz = armas_x_dot(&r, &z, cf);
        alpha = dot_rz / dot_p;

        // x = x + alpha*p
        armas_x_axpy(x, alpha, &p, cf);
        // r = r - alpha*Ap
        armas_x_axpy(&r, -alpha, &Ap, cf);
        // z = M^-1*r
        armas_x_mcopy(&z, &r);
        //__pr_vec("z.0", &r, 10);
        M->precond(&z, M, &r, cf);
        //__pr_vec("z.1", &z, 10);
        
        dot_rz1 = armas_x_dot(&r, &z, cf);
        printf("%3d: stop %e, dot_p %e, dot_rz %e, dot_rz1 %e\n", niter, rstop, dot_p, dot_rz, dot_rz1);
        if (dot_rz1 < rstop) {
            break;
        }
        beta = dot_rz1 / dot_rz;
        // p = beta*p + z;
        armas_x_scale_plus(beta, &p, __ONE, &z, 0, cf);
    }
    return niter;
}

coo_elem_t Clower[8] = {
    { .i = 0, .j = 0, .val = 4},
    { .i = 1, .j = 1, .val = 4},
    { .i = 2, .j = 2, .val = 4},
    { .i = 3, .j = 3, .val = 4},
    { .i = 4, .j = 4, .val = 10},
    { .i = 5, .j = 5, .val = 10},
    { .i = 4, .j = 0, .val = 2},
    { .i = 5, .j = 1, .val = 2}
};

int test_pcg(armas_d_sparse_t *A, int flags, int verbose,
             armassp_type_enum kind, int maxiter, DTYPE rstop, DTYPE rmult)
{
    armas_d_sparse_t *C, *Cu, *M, BL;
    armas_d_dense_t x, y, z, Cd;
    armassp_d_precond_t P;
    double relerr, nrm;
    DTYPE rd;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    armas_wbuf_t wb;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *uplo = (flags & ARMAS_UPPER) != 0 ? "U" : "L";

    BL = (armas_d_sparse_t){ .elems.ep = Clower, .rows = 6, .cols = 6, .nnz = 8, .kind = ARMASSP_COO};
    if (!A)
        A = &BL;

    
    armas_walloc(&wb, 5*A->rows*sizeof(DTYPE));
    
    C = armassp_d_convert(A, kind);
    M = armassp_d_convert(A, kind);

    // x and y 
    armas_d_init(&x, C->rows, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_init(&z, C->rows, 1);
    armas_d_set_values(&x, one, 0);
    
    armas_d_init(&Cd, C->rows, C->cols);
    
    armassp_d_init_icholz(&P, M, flags);
    
    armassp_d_todense(&Cd, C, &cf);
    armassp_d_mvmult_sym(0.0, &y,  1.0, C, &x, flags, &cf);

    armassp_params_t ipar = { .maxiter = maxiter, .stop = rstop, .tolmult = rmult };
    
    if (verbose > 1) {
        __x_pcgrad(&z, C, &y, &P, flags, maxiter, rmult, rstop, &rd, &wb, &cf);
        __pr_vec("y = A*x  ", &y, 10);
        __pr_vec("z = A-1*z", &z, 10);
    } else {
        armassp_x_pcgrad_w(&z, C, &y, &P, flags, &ipar, &wb, &cf);
        fprintf(stderr, "numiters = %d, res = %e, [tolmult=%e, rstop=%e]\n",
                ipar.numiters, ipar.residual, ipar.tolmult, ipar.stop);
    }

    relerr = rel_error(&nrm, &z, &x, ARMAS_NORM_INF, 0, &cf);

    fprintf(stderr, "%s (uplo=%s) ||x - A^-1*(A*x)||: %e\n", st, uplo, relerr);
    if (verbose) {
        armas_d_dense_t tmp;
        fprintf(stderr, "x - A^-1*(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
    }

    armas_wrelease(&wb);
    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armassp_d_free(C);
    armassp_d_free(M);
    return stat;
}


int test_precond(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind)
{
    armas_d_sparse_t *C, *Cu, *M, BL;
    armas_d_dense_t x, y, z, Cd, tmp;
    armassp_d_precond_t P;
    armas_conf_t cf = *armas_conf_default();

    BL = (armas_d_sparse_t){ .elems.ep = Clower, .rows = 6, .cols = 6, .nnz = 8, .kind = ARMASSP_COO};
    if (!A)
        A = &BL;
    
    C = armassp_d_convert(A, kind);
    M = armassp_d_convert(A, kind);

    // x and y 
    armas_d_init(&x, C->rows, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_init(&z, C->rows, 1);
    armas_d_set_values(&x, one, 0);

    // y = A*x; z = y
    armassp_d_mvmult_sym(0.0, &y,  1.0, C, &x, flags, &cf);
    //armas_d_mcopy(&z, &y);
    printf("     z: "); armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &y));
    
    armassp_d_init_icholz(&P, M, flags);
    P.precond(&z, &P, &y, 0);
    printf("P.-1*z: "); armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
}

int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = (char *)0;
    FILE *fp;
    int tc;
    int maxiter;
    DTYPE stop = 0.0;
    DTYPE rmult = 0.0;
    armas_d_sparse_t *A;

    while ((opt = getopt(argc, argv, "vf:n:R:M:")) != -1) {
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
    
    A = (armas_d_sparse_t *)0;
    if (path) {
        if (verbose > 0)
            fprintf(stderr, "opening '%s'...\n", path);

        if (! (fp = fopen(path, "r"))) {
            perror(path);
            exit(1);
        }
        A = armassp_d_mmload(&tc, fp);
    }

    test_precond(A, ARMAS_LOWER, verbose, ARMASSP_CSC);
    test_pcg(A, ARMAS_LOWER, verbose, ARMASSP_CSC, maxiter, stop, rmult);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
