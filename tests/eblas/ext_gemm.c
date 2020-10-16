
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include "testing.h"

int test_unit_b(int M, int N, int K, int verbose, int lb, armas_conf_t *cf)
{
    int fails = 0;
    armas_dense_t A, At, B, Bt, C, Ct, T, tcol, tc;
    int ok;
    DTYPE n0, n1;
    armas_env_t *env = armas_getenv();
    env->lb = lb;
    armas_init(&C, M, N);
    armas_init(&Ct, N, M);
    armas_init(&T, M, N);
    armas_set_values(&C, zero, 0);
    armas_set_values(&Ct, zero, 0);

    armas_init(&A, M, K);
    armas_init(&At, K, M);
    armas_init(&B, K, N);
    armas_init(&Bt, N, K);

    armas_column(&tcol, &T, 0);
    make_ext_matrix_data(&A, 1.0, &tcol, ARMAS_LEFT);
    for (int i = 1; i < N; i++) {
        armas_column(&tc, &T, i);
        armas_copy(&tc, &tcol, cf);
    }
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    armas_set_values(&B, one, 0);
    armas_set_values(&Bt, one, 0);

    // ------------------------------------------------------------
    //  C = A*B
    armas_ext_mult(ZERO, &C, ONE, &A, &B, 0, cf);
    armas_mcopy(&Ct, &C, ARMAS_TRANS, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_gemm(A, B)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // ------------------------------------------------------------
    //  C^T = B^T*A^T
    armas_ext_mult(ZERO, &Ct, ONE, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &Ct, &T, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_gemm(B^T, A^T)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // ------------------------------------------------------------
    //  C = A^T*B
    armas_ext_mult(ZERO, &C, ONE, &At, &B, ARMAS_TRANSA, cf);
    if (N < 10 && verbose > 2) {
        printf("ext: A*B\n"); armas_printf(stdout, "%9.2e", &C);
    }
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_INF, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_gemm(A^T, B)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // ------------------------------------------------------------
    //  C^T = B^T*A
    armas_ext_mult(ZERO, &Ct, ONE, &B, &At, ARMAS_TRANSA, cf);
    n0 = rel_error(&n1, &Ct, &T, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: expected == ext_gemm(B^T, A)\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_release(&A);
    armas_release(&At);
    armas_release(&B);
    armas_release(&Bt);
    armas_release(&C);
    armas_release(&Ct);
    armas_release(&T);
    return fails;
}

int test_almost_one(int M, int N, int K, int verbose, int lb, armas_conf_t *cf)
{
    armas_dense_t A, At, B, Bt, C, Ct, T;
    int fails = 0;
    int ok;
    DTYPE n0, n1;
    armas_env_t *env = armas_getenv();
    env->lb = lb;
    armas_init(&C, M, N);
    armas_init(&Ct, N, M);
    armas_init(&T, M, 1);
    armas_set_values(&C, zero, 0);
    armas_set_values(&Ct, zero, 0);

    armas_init(&A, M, K);
    armas_init(&At, K, M);
    armas_init(&B, K, N);
    armas_init(&Bt, N, K);

    make_ext_matrix_data(&A, 1.0, &T, ARMAS_LEFT);
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    armas_set_values(&B, almost_one, 0);
    armas_mcopy(&Bt, &B, ARMAS_TRANS, cf);
    if (verbose > 2) {
        MAT_PRINT("B", &B);
    }
    // ------------------------------------------------------------
    //  C = A*B
    armas_ext_mult(ZERO, &C, ONE, &A, &B, 0, cf);
    //  C^T = B^T*A^T
    armas_ext_mult(ZERO, &Ct, ONE, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &Ct, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: (A*B)^T == B^T*A^T\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // ------------------------------------------------------------
    //  C = A^T*B
    armas_ext_mult(ZERO, &C, ONE, &At, &B, ARMAS_TRANSA, cf);
    //  C^T = B^T*A
    armas_ext_mult(ZERO, &Ct, ONE, &Bt, &A, ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &Ct, ARMAS_NORM_INF, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: (A^T*B)^T == B^T*A\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_release(&A);
    armas_release(&At);
    armas_release(&B);
    armas_release(&Bt);
    armas_release(&C);
    armas_release(&Ct);
    armas_release(&T);
    return fails;
}


int main(int argc, char **argv)
{

    armas_conf_t conf;

    int opt;
    int N = 33;
    int M = 33;
    int K = 33;
    int LB = 0;
    int fails = 0;
    int verbose = 0;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: ext_gemm -v [size]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
        M = K = N;
    }

    conf = *armas_conf_default();

    fails += test_unit_b(M, N, K, verbose, LB, &conf);
    fails += test_almost_one(M, N, K, verbose, LB, &conf);

    exit(fails);
}
