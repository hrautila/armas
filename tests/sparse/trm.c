
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

// C = op(A)*op(A)
int test_trm(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";
    char *uplo = (flags & ARMAS_UPPER) != 0 ? "U" : "L";

    C = armassp_d_convert(A, kind);
    // x and y 
    armas_d_init(&x, C->rows, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_set_values(&x, one, 0);
    armas_d_set_values(&y, one, 0);
    
    if ((flags & ARMAS_UPPER) != 0) {
        Cu = armassp_d_transpose(C);
        armassp_d_mvmult_trm(&y,  1.0, Cu, flags, &cf);
        armassp_d_mvsolve_trm(&y, 1.0, Cu, flags, &cf);
        armassp_d_free(Cu);
    } else {
        armassp_d_mvmult_trm(&y,  1.0, C, flags, &cf);
        armassp_d_mvsolve_trm(&y, 1.0, C, flags, &cf);
    }

    relerr = rel_error(&nrm, &x, &y, ARMAS_NORM_INF, 0, &cf);

    fprintf(stderr, "%s (uplo=%s, trans=%s) ||x - A^-1*(A*x)||: %e\n", st, uplo, sf, relerr);
    if (verbose) {
        armas_d_dense_t tmp;
        fprintf(stderr, "x - A^-1*(A*x):\n");
        armas_d_printf(stderr, "%.2e", armas_d_col_as_row(&tmp, &x));
    }

    armas_d_release(&x);
    armas_d_release(&y);
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


    test_trm(A, ARMAS_LOWER, verbose, ARMASSP_CSR);
    test_trm(A, ARMAS_LOWER|ARMAS_TRANS, verbose, ARMASSP_CSR);
    test_trm(A, ARMAS_LOWER, verbose, ARMASSP_CSC);
    test_trm(A, ARMAS_LOWER|ARMAS_TRANS, verbose, ARMASSP_CSC);

    test_trm(A, ARMAS_UPPER, verbose, ARMASSP_CSR);
    test_trm(A, ARMAS_UPPER|ARMAS_TRANS, verbose, ARMASSP_CSR);
    test_trm(A, ARMAS_UPPER, verbose, ARMASSP_CSC);
    test_trm(A, ARMAS_UPPER|ARMAS_TRANS, verbose, ARMASSP_CSC);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
