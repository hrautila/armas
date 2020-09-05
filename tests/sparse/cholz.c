
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"
#include <assert.h>

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

coo_elem_t Clower2[9] = {
    { .i = 0, .j = 0, .val = 4},
    { .i = 1, .j = 1, .val = 4},
    { .i = 2, .j = 2, .val = 4},
    { .i = 3, .j = 3, .val = 4},
    { .i = 4, .j = 4, .val = 10},
    { .i = 5, .j = 5, .val = 10},
    { .i = 3, .j = 0, .val = 2},
    { .i = 4, .j = 0, .val = 2},
    { .i = 5, .j = 1, .val = 2}
};

coo_elem_t Cupper[8] = {
    { .i = 0, .j = 0, .val = 4},
    { .i = 1, .j = 1, .val = 4},
    { .i = 2, .j = 2, .val = 4},
    { .i = 3, .j = 3, .val = 4},
    { .i = 4, .j = 4, .val = 10},
    { .i = 5, .j = 5, .val = 10},
    { .i = 0, .j = 4, .val = 2},
    { .i = 1, .j = 5, .val = 2}
};

#define null_dense (armas_d_dense_t *)0


int test_carray(int verbose)
{
    armas_d_sparse_t C, *A, *L;
    armas_d_dense_t Cd, Ld;
    armas_conf_t *cf = armas_conf_default();
    int flags = ARMAS_LOWER;
    armassp_type_enum kind = ARMASSP_CSR;
    
    C = (armas_d_sparse_t){ .elems.ep = Clower2, .rows = 6, .cols = 6, .nnz = 9, .kind = ARMASSP_COO};
    A = armassp_d_convert(&C, kind);
    L = armassp_d_mkcopy(A);
    
    armas_d_init(&Cd, C.rows, C.cols);
    armas_d_init(&Ld, C.rows, C.cols);
    armassp_d_todense(&Cd, A, cf);
    printf("C\n"); armas_d_printf(stdout, "%.2f", &Cd);

    armas_d_cholfactor(&Cd, null_dense, ARMAS_NOPIVOT, flags, cf);
    printf("LL^T\n"); armas_d_printf(stdout, "%.2f", &Cd);
      
    armassp_d_icholz(L, ARMAS_LOWER);
    armassp_d_todense(&Ld, L, cf);
    printf("sp(LL^T)\n"); armas_d_printf(stdout, "%.2f", &Ld);
}

// dense(A)*x - sparse(A)*x
int test_icholz(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t Cd, Cd2, *W;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";
    char *uplo = (flags & ARMAS_UPPER) != 0 ? "U" : "L";

    W = (armas_d_dense_t *)0;
    
    C = armassp_d_convert(A, kind);
    // x and y 
    
    armas_d_init(&Cd, C->rows, C->cols);
    armas_d_init(&Cd2, C->rows, C->cols);
    
    if ((flags & ARMAS_UPPER) != 0) {
        Cu = armassp_d_transpose(C);
        armassp_d_free(C);
        C = Cu;
    } 
    armassp_d_todense(&Cd, C, &cf);

    
    armassp_d_icholz(C, flags);
    armassp_d_todense(&Cd2, C, &cf);

    armas_d_cholfactor(&Cd, W, ARMAS_NOPIVOT, flags, &cf);
    
    if (verbose > 1) {
        fprintf(stderr, "icholz(C)\n");
        armas_d_printf(stderr, "%9.2e", &Cd2);
        fprintf(stderr, "chol(C)\n");
        armas_d_printf(stderr, "%9.2e", &Cd);
    }
                    
    //relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_INF, 0, &cf);

    armas_d_release(&Cd);
    armas_d_release(&Cd2);
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


    test_carray(verbose);
    //test_icholz(A, ARMAS_LOWER, verbose, ARMASSP_CSR);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSR);
    //test_sym(A, ARMAS_LOWER, verbose, ARMASSP_CSC);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSC);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
