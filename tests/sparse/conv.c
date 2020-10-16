
// Copyright by libARMAS authors. See AUTHORS file in this archive.

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

void nzprint(FILE *f, const char *frm, double *x, int n)
{ 
    fprintf(f, "[");
    for (int i = 0; i < n; i++) {
        if (i > 0)
            fprintf(f, ", ");
        fprintf(f, frm, x[i]);
    }
    fprintf(f, "]\n");
}

int test_convert(armas_d_sparse_t *A, armassp_type_enum kind, int verbose)
{
    armas_d_sparse_t *C, *C1;
    armas_d_dense_t Cd, Ad;
    armas_conf_t *cf = armas_conf_default();
    char *typename = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *t2 = kind == ARMASSP_CSR ? "CSC" : "CSR";

    // 1. convert A to request compressed matrix type
    // 2. convert both to dense
    // 3. compute difference; expect zero
    armas_d_init(&Ad, A->rows, A->cols);
    armassp_d_todense(&Ad, A, cf);
    if (verbose > 1) {
        fprintf(stderr, "A:\n");
        armas_d_printf(stderr, "%4.2f", &Ad);
    }

    C = armassp_d_convert(A, kind);
    armas_d_init(&Cd, A->rows, A->cols);
    armassp_d_todense(&Cd, C, cf);
    if (verbose > 1) {
        fprintf(stderr, "C:\n");
        armas_d_printf(stderr, "%4.2f", &Cd);
    }

    double rel_err, nrm2;
    rel_err = rel_error(&nrm2, &Cd, &Ad, ARMAS_NORM_INF, 0, cf);

    int ok = rel_err == 0.0;
    int stat = ok;
    fprintf(stderr, "COO->%s [%d, %d] %d non-zeros (%.3e %s)\n", typename, C->rows, C->cols, C->nnz, rel_err, PASS(ok));
    if (verbose > 0)
        armassp_d_iprintf(stderr, C);
    if (verbose > 1)
        nzprint(stderr, "%4.2f", C->elems.v, C->nnz);

    C1 = armassp_d_convert(C, kind == ARMASSP_CSR ? ARMASSP_CSC : ARMASSP_CSR);
    armassp_d_todense(&Cd, C1, cf);
    rel_err = rel_error(&nrm2, &Cd, &Ad, ARMAS_NORM_INF, 0, cf);

    ok = rel_err == 0.0;
    stat = stat && ok;
    fprintf(stderr, "%s->%s [%d, %d] %d non-zeros (%.3e %s)\n", typename, t2, C1->rows, C1->cols, C1->nnz, rel_err, PASS(ok));
    if (verbose > 0)
        armassp_d_iprintf(stderr, C1);

    armassp_d_free(C);
    armassp_d_free(C1);
    armas_d_release(&Ad);
    armas_d_release(&Cd);
    return stat;
}


int test_transpose(armas_d_sparse_t *A, armassp_type_enum kind, int verbose)
{
    armas_d_sparse_t *C, *C1, *C2;
    armas_d_dense_t Cd, C1d, C2d;
    armas_conf_t *cf = armas_conf_default();
    char *typename = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *t2 = kind == ARMASSP_CSR ? "CSC" : "CSR";

    // 1. convert A to request compressed matrix type
    // 2. convert both to dense
    // 3. compute difference; expect zero

    C = armassp_d_convert(A, kind);
    armas_d_init(&Cd, A->rows, A->cols);
    armassp_d_todense(&Cd, C, cf);

    C1 = armassp_d_transpose(C);
    armas_d_init(&C1d, C1->rows, C1->cols);
    armassp_d_todense(&C1d, C1, cf);

    double rel_err, nrm2;
    rel_err = rel_error(&nrm2, &C1d, &Cd, ARMAS_NORM_INF, ARMAS_TRANS, cf);

    int ok = rel_err == 0.0;
    int stat = ok;
    fprintf(stderr, "%s [%d,%d] -> [%d,%d] %d non-zeros (%.3e %s)\n",
            typename, C->rows, C->cols, C1->rows, C1->cols, C1->nnz, rel_err, PASS(ok));

    if (verbose > 0) {
        fprintf(stderr, "C:\n"); armassp_d_iprintf(stderr, C);
    }
    if (verbose > 0) {
        fprintf(stderr, "C.T:\n"); armassp_d_iprintf(stderr, C1);
    }

    armassp_d_free(C);
    armassp_d_free(C1);
    armas_d_release(&C1d);
    armas_d_release(&Cd);
    return stat;
}

int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = "test1.mtx";
    FILE *fp;
    int tc;
    int testno = 0;
    armas_d_sparse_t *A;

    while ((opt = getopt(argc, argv, "vf:t:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'f':
            path = optarg;
            break;
        case 't':
            testno = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: conv [-v -t testno -f path] \n");
            fprintf(stderr, "  1 : COO->CSR; CSR->CSC\n");
            fprintf(stderr, "  2 : COO->CSC; CSC->CSR\n");
            fprintf(stderr, "  3 : COO->CSR; TRANSPOSE\n");
            fprintf(stderr, "  4 : COO->CSC; TRANSPOSE\n");
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
    fprintf(stderr, "type bits: x%0x\n", tc);
    if (testno == 0 || testno == 1)
        test_convert(A, ARMASSP_CSR, verbose);   
    if (testno == 0 || testno == 2)
        test_convert(A, ARMASSP_CSC, verbose);

    if (testno == 0 || testno == 3)
        test_transpose(A, ARMASSP_CSR, verbose);   
    if (testno == 0 || testno == 4)
        test_transpose(A, ARMASSP_CSC, verbose);   
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
