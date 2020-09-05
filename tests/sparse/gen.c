
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

// dense(A)*x - sparse(A)*x
int test_gen(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y, z, Cd;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";
    char *uplo = (flags & ARMAS_UPPER) != 0 ? "U" : "L";
    int m = (flags & ARMAS_TRANS) == 0 ? A->rows : A->cols;
    int n = (flags & ARMAS_TRANS) == 0 ? A->cols : A->rows;
    
    // x and y 
    armas_d_init(&x, n, 1);
    armas_d_init(&y, m, 1);
    armas_d_init(&z, m, 1);
    armas_d_set_values(&x, one, 0);
    
    C = armassp_d_convert(A, kind);
    armas_d_init(&Cd, C->rows, C->cols);
    
    armassp_d_todense(&Cd, C, &cf);
    armassp_d_mvmult(0.0, &y,  1.0, C, &x, flags, &cf);

    armas_d_mvmult(0.0, &z, 1.0, &Cd, &x, flags, &cf);
    if (verbose > 1) {
        armas_d_dense_t tmp;
        fprintf(stderr, "dense(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
        fprintf(stderr, "sparse(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &y));
    }

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_INF, 0, &cf);

    fprintf(stderr, "%s (trans=%s) ||dense(A)*x - sparse(A)*x||: %e\n", st, sf, relerr);
    if (verbose) {
        armas_d_dense_t tmp;
        fprintf(stderr, "dense(A*x) - sparse(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &z));
    }

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&Cd);
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

    while ((opt = getopt(argc, argv, "vf:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'f':
            path = optarg;
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


    test_gen(A, 0, verbose, ARMASSP_CSR);
    test_gen(A, ARMAS_TRANS, verbose, ARMASSP_CSR);
    test_gen(A, 0, verbose, ARMASSP_CSC);
    test_gen(A, ARMAS_TRANS, verbose, ARMASSP_CSC);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
