
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"

int test_unit_b(int M, int N, int verbose, int flags, armas_conf_t *cf)
{
    int fails = 0;
    armas_x_dense_t A, B, Bt, C, Ct, T, tcol, tc;
    int ok;
    DTYPE n0, n1;
    const char *uplo = (flags &ARMAS_UPPER) ? "upper" : "lower";

    armas_x_init(&C, M, N);
    armas_x_init(&Ct, N, M);
    armas_x_init(&T, M, N);
    armas_x_set_values(&C, zero, 0);
    armas_x_set_values(&Ct, zero, 0);

    armas_x_init(&A, M, M);
    armas_x_init(&B, M, N);
    armas_x_init(&Bt, N, M);

    armas_x_row(&tcol, &T, 0);
    make_ext_matrix_data(&B, 1.0, &tcol, ARMAS_RIGHT);
    for (int i = 1; i < M; i++) {
        armas_x_row(&tc, &T, i);
        armas_x_copy(&tc, &tcol, cf);
    }
    armas_x_mcopy(&Bt, &B, ARMAS_TRANS, cf);
    armas_x_set_values(&A, one, 0);
    if (verbose > 2) {
        MAT_PRINT("T", &T);
        MAT_PRINT("B", &B);
        MAT_PRINT("B^T", &Bt);
    }

    // ------------------------------------------------------------
    //  C = A*B
    armas_x_ext_mult_sym(ZERO, &C, ONE, &A, &B, flags|ARMAS_LEFT, cf);
    if (verbose > 2) {
        MAT_PRINT("A*B", &C);
    }
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_INF, 0, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_symm(%s(A), B)\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    // ------------------------------------------------------------
    //  C = A*B^T
    armas_x_ext_mult_sym(ZERO, &C, ONE, &A, &Bt, flags|ARMAS_LEFT|ARMAS_TRANSB, cf);
    if (verbose > 2) {
        MAT_PRINT("A*B^T", &C);
    }
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_INF, 0, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_symm(%s(A), B^T)\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    // ------------------------------------------------------------
    //  C = B*A
    armas_x_ext_mult_sym(ZERO, &Ct, ONE, &A, &Bt, flags|ARMAS_RIGHT, cf);
    if (verbose > 2) {
        MAT_PRINT("B*A", &Ct);
    }
    n0 = rel_error(&n1, &Ct, &T, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_symm(B, %s(A))\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    // ------------------------------------------------------------
    //  C^T = B^T*A
    armas_x_ext_mult_sym(ZERO, &Ct, ONE, &A, &B, flags|ARMAS_RIGHT|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &Ct, &T, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_symm(B^T, %s(A))\n", PASS(ok), uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_x_release(&A);
    armas_x_release(&B);
    armas_x_release(&Bt);
    armas_x_release(&C);
    armas_x_release(&Ct);
    armas_x_release(&T);
    return fails;
}

int test_almost_one(int M, int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_x_dense_t A, B, Bt, C, Ct, tc;
    int fails = 0;
    int ok;
    DTYPE n0, n1;
    const char *uplo = (flags & ARMAS_UPPER) ? "upper" : "lower";

    armas_x_init(&C, M, N);
    armas_x_init(&Ct, N, M);
    armas_x_set_values(&C, zero, 0);
    armas_x_set_values(&Ct, zero, 0);

    armas_x_init(&A, M, M);
    armas_x_init(&B, M, N);
    armas_x_init(&Bt, N, M);
    armas_x_init(&tc, M, 1);

    make_ext_matrix_data(&B, 1.0, &tc, ARMAS_RIGHT);
    armas_x_mcopy(&Bt, &B, ARMAS_TRANS, cf);
    armas_x_set_values(&A, almost_one, ARMAS_SYMM);
    if (verbose > 2) {
        MAT_PRINT("A", &A);
        MAT_PRINT("B", &B);
        MAT_PRINT("B^T", &Bt);
    }

    // ------------------------------------------------------------
    //  C = A*B
    armas_x_ext_mult_sym(ZERO, &C, ONE, &A, &B, flags|ARMAS_LEFT, cf);
    //  C^T = B^T*A
    armas_x_ext_mult_sym(ZERO, &Ct, ONE, &A, &Bt, flags|ARMAS_RIGHT, cf);
    n0 = rel_error(&n1, &C, &Ct, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: (%s(A)*B)^T == B*%s(A)\n", PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // ------------------------------------------------------------
    //  C = A*B^T
    armas_x_ext_mult_sym(ZERO, &C, ONE, &A, &Bt, flags|ARMAS_LEFT|ARMAS_TRANSB, cf);
    armas_x_ext_mult_sym(ZERO, &Ct, ONE, &A, &B, flags|ARMAS_RIGHT|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &Ct, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = (n0 == 0.0 || isOK(n0, N)) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: (%s(A)*B^T)^T == B^T*%s(A)\n", PASS(ok), uplo, uplo);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_x_release(&A);
    armas_x_release(&B);
    armas_x_release(&Bt);
    armas_x_release(&C);
    armas_x_release(&Ct);
    armas_x_release(&tc);
    return fails;
}


int main(int argc, char **argv)
{

    armas_conf_t conf;

    int opt;
    int N = 111;
    int M = 121;
    int fails = 0;
    int verbose = 0;
    int doall = 1;
    int upper = 0;
    int lower = 0;

    while ((opt = getopt(argc, argv, "vLU")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'U':
            doall = 0;
            upper = 1;
            break;
        case 'L':
            doall = 0;
            lower = 1;
            break;
        default:
            fprintf(stderr, "usage: ext_gemm -v [size]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    }

    conf = *armas_conf_default();

    if (doall) {
        fails += test_unit_b(M, N, verbose, ARMAS_UPPER, &conf);
        fails += test_unit_b(M, N, verbose, ARMAS_LOWER, &conf);
        fails += test_almost_one(M, N, verbose, ARMAS_UPPER, &conf);
        fails += test_almost_one(M, N, verbose, ARMAS_LOWER, &conf);
    } else {
        if (upper)
            fails += test_unit_b(M, N, verbose, ARMAS_UPPER, &conf);
        if (lower)
            fails += test_unit_b(M, N, verbose, ARMAS_LOWER, &conf);
        if (upper)
            fails += test_almost_one(M, N, verbose, ARMAS_UPPER, &conf);
        if (lower)
            fails += test_almost_one(M, N, verbose, ARMAS_LOWER, &conf);
    }

    exit(fails);
}
