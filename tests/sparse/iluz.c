
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"

/*  sparseLU == denseLU
    4 . . . . .
    . 4 . . . 2
    . . 4 . . .
    . . . 4 . .
    8 9 . . 4 4
    . 9 . . . 4
 */

coo_elem_t Cdata[11] = {
    { .i = 0, .j = 0, .val = 4},
    { .i = 1, .j = 1, .val = 4},
    { .i = 2, .j = 2, .val = 4},
    { .i = 3, .j = 3, .val = 4},
    { .i = 4, .j = 4, .val = 4},
    { .i = 5, .j = 5, .val = 4},
    { .i = 1, .j = 5, .val = 2},
    { .i = 4, .j = 0, .val = 8},
    { .i = 4, .j = 1, .val = 9},
    { .i = 4, .j = 5, .val = 4},
    { .i = 5, .j = 1, .val = 9}
};

int __ilufactor_nopiv(armas_d_dense_t *A, armas_d_sparse_t *As, armas_conf_t *conf)
{
    armas_d_dense_t a12, a21, A22, ABR;
    double a11val;
    int j, k;

    for (int i = 0; i < A->rows; i++) {
        // a21 = a21/a11
        a11val = armas_x_get(A, i, i);
        armas_d_submatrix(&a21, A, i+1, i, A->rows-i-1, 1);
        armas_d_scale(&a21, 1.0/a11val, conf);

        armas_d_submatrix(&a12, A, i, i+1, 1, A->rows-i-1);
        armas_d_submatrix(&A22, A, i+1, i+1, A->rows-i-1, A->rows-i-1);
        // A22 = A22 - a21*a12
        armas_d_mvupdate(&A22, -1.0, &a21, &a12, conf);

        // Zero elements not part of sparse matrix; 
        if (As->kind == ARMASSP_CSC) {
            for (j = 0; j < A22.cols; j++) {
                for (k = 0; k < A22.rows; k++) {
                    if (armassp_d_nz(As, i+j, i+k) < 0)
                        armas_d_set(&A22, k, j, 0.0);
                }
            }
        } else {
            for (k = 0; k < A22.rows; k++) {
                for (j = 0; j < A22.cols; j++) {
                    if (armassp_d_nz(As, i+k, i+j) < 0)
                        armas_d_set(&A22, k, j, 0.0);
                }
            }
        }
    }
    return 0;
}

int iprintf(FILE *f, armas_d_dense_t *A, int order)
{
    int nc;
    if (order == ARMASSP_CSR) {
        for (int i = 0; i < A->rows; i++) {
            fprintf(f, "r%d: [", i);
            nc = 0;
            for (int j = 0; j < A->cols; j++) {
                if (armas_d_get_unsafe(A, i, j) == 0.0)
                    continue;
                if (nc > 0)
                    fprintf(f, ",");
                fprintf(f, "%d", j);
                nc++;
            }
            fprintf(f, "]\n");
        }
    }
    else {
        for (int j = 0; j < A->cols; j++) {
            fprintf(f, "c%d: [", j);
            nc = 0;
            for (int i = 0; i < A->rows; i++) {
                if (armas_d_get_unsafe(A, i, j) == 0.0)
                    continue;
                if (nc > 0)
                    fprintf(f, ",");
                fprintf(f, "%d", i);
                nc++;
            }
            fprintf(f, "]\n");
        }
    }
}

int isprintf(FILE *f, armas_d_sparse_t *A)
{
    int nc;
    if (A->kind == ARMASSP_CSR) {
        for (int k = 0; k < A->rows; k++) {
            fprintf(f, "r%d: [", k);
            nc = 0;
            for (int p = A->ptr[k]; p < A->ptr[k+1]; p++) {
                if (A->elems.v[p] == 0.0)
                    continue;
                if (nc > 0)
                    fprintf(f, ",");
                fprintf(f, "%d", A->ix[p]);
                nc++;
            }
            fprintf(f, "]\n");
        }
    }
    else {
        for (int k = 0; k < A->cols; k++) {
            fprintf(f, "c%d: [", k);
            nc = 0;
            for (int p = A->ptr[k]; p < A->ptr[k+1]; p++) {
                if (A->elems.v[p] == 0.0)
                    continue;
                if (nc > 0)
                    fprintf(f, ",");
                fprintf(f, "%d", A->ix[p]);
                nc++;
            }
            fprintf(f, "]\n");
        }
    }
}

int compress_nz(armas_d_sparse_t *A)
{
    int i, p0, p1, *Ai;
    double *Ae;
    
    if (A->kind != ARMASSP_CSC && A->kind != ARMASSP_CSR)
        return -1;
    
    p0 = 0;
    Ae = A->elems.v;
    Ai = A->ix;
    for (i = 0; i < A->nptr; i++) {
        p1 = p0;
        for (int p = A->ptr[i]; p < A->ptr[i+1]; p++) {
            if (Ae[p] != 0.0) {
                Ae[p0] = Ae[p];
                Ai[p0] = Ai[p];
                p0++;
            }
        }
        A->ptr[i] = p1;
    }
    A->ptr[i] = p0;
    A->nnz = p0;
    return 0;
}

// 
int test_iluz(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind)
{
    armas_d_sparse_t *C, *Cu, B;
    armas_d_dense_t x, y, z, Cd, Cd2, *W, r;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    char *st = kind == ARMASSP_CSR ? "CSR" : "CSC";
    char *sf = (flags & ARMAS_TRANS) != 0 ? "T" : "N";

    W = (armas_d_dense_t *)0;
    
    B = (armas_d_sparse_t){
        .elems.ep = Cdata,
        .rows = 6, .cols = 6, .nnz = 11,
        .kind = ARMASSP_COO};

    if (!A)
        A = &B;
    
    //armassp_d_sort_to(A, kind == ARMASSP_CSR ? ARMASSP_ROW_ORDER : ARMASSP_COL_ORDER);
    C = armassp_d_convert(A, kind);  

    // x and y 
    armas_d_init(&x, C->rows, 1);
    armas_d_init(&y, C->rows, 1);
    armas_d_init(&z, C->rows, 1);
    armas_d_set_values(&x, one, 0);
    armas_d_set_values(&z, one, 0);
    // y = A*x
    armassp_d_mvmult(0.0, &y, 1.0, C, &x, 0, &cf);
    printf("     y = Ax : ");
    armas_d_printf(stdout, "%9.2e", armas_d_col_as_row(&r, &y));
    

    armas_d_init(&Cd, C->rows, C->cols);
    armas_d_init(&Cd2, C->rows, C->cols);
    
    armassp_d_todense(&Cd, C, &cf);

    if (verbose > 1) {
        printf("C\n");
        armas_d_printf(stdout, "%6.2f", &Cd);
        printf("sparse(C)\n");
        isprintf(stdout, C);
    }
    armas_d_lufactor(&Cd, ARMAS_NOPIVOT, &cf);
    if (verbose > 1) {
        printf("LU(C)\n");
        armas_d_printf(stdout, "%6.2f", &Cd);
    }
    
    // incomplete LU factorization
    armassp_d_iluz(C);  
    if (verbose > 1) {
        armassp_d_todense(&Cd2, C, &cf);
        printf("ILU(C)\n");
        armas_d_printf(stdout, "%6.2f", &Cd2);
    }
  
    if (A == &B) {
        // sparseLU == denseLU; verify
        armassp_d_mvmult_trm(&z, 1.0, C, ARMAS_UPPER|ARMAS_LEFT, &cf);
        armassp_d_mvmult_trm(&z, 1.0, C, ARMAS_LOWER|ARMAS_LEFT|ARMAS_UNIT, &cf);

        printf("      L(Ux) : ");
        armas_d_printf(stdout, "%9.2e", armas_d_col_as_row(&r, &z));

        armas_d_axpy(&y, -1.0, &z, &cf);
        printf("  y - L(Ux) : ");
        armas_d_printf(stdout, "%9.2e", armas_d_col_as_row(&r, &y));
        // A^-1*z = (LU)^-1*z = U^-1*(L^-1*z)
        armassp_d_mvsolve_trm(&z, 1.0, C, ARMAS_LOWER|ARMAS_LEFT|ARMAS_UNIT, &cf);
        armassp_d_mvsolve_trm(&z, 1.0, C, ARMAS_UPPER|ARMAS_LEFT, &cf);
        printf("U^-1(L^-1 x): ");
        armas_d_printf(stdout, "%9.2e", armas_d_col_as_row(&r, &z));
    }
#if 0
    if (verbose > 1) {
        fprintf(stderr, "sparse(C)\n");
        //armassp_d_iprintf(stderr, C);
        isprintf(stderr, C);
        //armas_d_printf(stderr, "%9.2e", &Cd2);
        fprintf(stderr, "lu(C)\n");
        iprintf(stderr, &Cd, C->kind);
        //armas_d_printf(stderr, "%9.2e", &Cd);
    }
#endif
    
    //relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_INF, 0, &cf);

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&Cd);
    armas_d_release(&Cd2);
    armassp_d_free(C);
    return stat;
}

int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = (char *)0;
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

    test_iluz(A, 0, verbose, ARMASSP_CSR);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSR);
    //test_sym(A, ARMAS_LOWER, verbose, ARMASSP_CSC);
    //test_sym(A, ARMAS_UPPER, verbose, ARMASSP_CSC);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
