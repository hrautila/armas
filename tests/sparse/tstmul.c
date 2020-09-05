
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

// C = op(A)*op(A)
int test_mult_self(armas_d_sparse_t *A, armassp_type_enum kind)
{
    armas_d_accum_t acc;
    armas_d_sparse_t *C, *B;
    armas_d_dense_t Dc, Ac, Ab;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    
    C = armassp_d_convert(A, kind);
    armas_d_init(&Dc, C->rows, C->cols);
    armassp_d_todense(&Dc, C, &cf);
    
    B = armassp_d_mult(1.0, C, C, ARMAS_NONE, &cf);
    armas_d_init(&Ab, B->rows, B->cols);
    armassp_d_todense(&Ab, B, &cf);

    armas_d_mult(0.0, &Ac, 1.0, &Dc, &Dc, 0, &cf);

    relerr = rel_error(&nrm, &Ab, &Ac, ARMAS_NORM_INF, 0, &cf);
    armas_d_scale_plus(1.0, &Ab, -1.0, &Ac, 0, &cf);

    fprintf(stderr, "relerr: %e, nrm: %e\n", relerr, nrm);

    armas_d_release(&Dc);
    armas_d_release(&Ac);
    armas_d_release(&Ab);
    armassp_d_free(C);
    armassp_d_free(B);
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

    test_mult_self(A, ARMASSP_CSR);
    
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
