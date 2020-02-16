
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

#define ANULL (armas_x_dense_t *)0

/*
 * C = [M, N], A = [M, M], B = [M, N]
 *
 * test1:  symm(A, B) - gemm(A, B) == 0
 */
int test_left(int M, int N, int K, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t A, B, C, Bt, T;
    int ok, fails = 0;
    DTYPE n0, n1;

    armas_x_init(&C, M, N);
    armas_x_init(&T, M, N);
    armas_x_set_values(&C, zero, 0);

    // test 1: M != N != K
    armas_x_init(&A, M, M);
    armas_x_init(&B, M, N);
    armas_x_init(&Bt, N, M);
    armas_x_set_values(&A, unitrand, ARMAS_SYMM);
    armas_x_set_values(&B, unitrand, 0);
    armas_x_mcopy(&Bt, &B, ARMAS_TRANS, cf);

    armas_x_mult_sym(0.0, &C, 1.0, &A, &B, ARMAS_LEFT|ARMAS_UPPER, cf);
    armas_x_mult(0.0, &T, 1.0, &A, &B, 0, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(upper(A), B)   == gemm(A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mult_sym(0.0, &C, 1.0, &A, &Bt, ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSB, cf);
    armas_x_mult(0.0, &T, 1.0, &A, &Bt, ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(upper(A), B^T) == gemm(A, B^T))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mult_sym(0.0, &C, 1.0, &A, &B, ARMAS_LEFT|ARMAS_LOWER, cf);
    armas_x_mult(0.0, &T, 1.0, &A, &B, 0, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(lower(A), B)   == gemm(A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mult_sym(0.0, &C, 1.0, &A, &Bt, ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANSB, cf);
    armas_x_mult(0.0, &T, 1.0, &A, &Bt, ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(lower(A), B^T) == gemm(A, B^T))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int test_right(int M, int N, int K, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t A, B, C, Bt, T;
    int ok, fails = 0;
    DTYPE n0, n1;

    armas_x_init(&C, M, N);
    armas_x_init(&T, M, N);
    armas_x_set_values(&C, zero, 0);

    // test 1: M != N != K
    armas_x_init(&A, N, N);
    armas_x_init(&B, M, N);
    armas_x_init(&Bt, N, M);
    armas_x_set_values(&A, unitrand, ARMAS_SYMM);
    armas_x_set_values(&B, unitrand, 0);
    armas_x_mcopy(&Bt, &B, ARMAS_TRANS, cf);

    armas_x_mult_sym(0.0, &C, 1.0, &A, &B, ARMAS_RIGHT|ARMAS_UPPER, cf);
    armas_x_mult(0.0, &T, 1.0, &B, &A, 0, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(B, upper(A))   == gemm(B, A))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mult_sym(0.0, &C, 1.0, &A, &Bt, ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSB, cf);
    armas_x_mult(0.0, &T, 1.0, &Bt, &A, ARMAS_TRANSA, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(B^T, upper(A)) == gemm(B^T, A))\n", PASS(ok));
    fails += 1 - ok;

    armas_x_mult_sym(0.0, &C, 1.0, &A, &B, ARMAS_RIGHT|ARMAS_LOWER, cf);
    armas_x_mult(0.0, &T, 1.0, &B, &A, 0, cf);
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(B, lower(A))   == gemm(B, A))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mult_sym(0.0, &C, 1.0, &A, &Bt, ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSB, cf);
    armas_x_mult(0.0, &T, 1.0, &Bt, &A, ARMAS_TRANSA, cf);
    n0 = rel_error(&n1, &C, &T, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: symm(B^T, lower(A)) == gemm(B^T, A))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;
    return fails;
}


int main(int argc, char **argv)
{

    armas_conf_t conf;

    int opt;
    int N = 387;
    int M = 451;
    int K = 337;
    int verbose = 0;
    int fails = 0;
    conf = *armas_conf_default();
    armas_env_t *env = armas_getenv();
    env->fixed = 1;
    env->mb = env->nb = env->kb = 4;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        default:
            fprintf(stderr, "usage: test_symm [-v] [size]\n");
            exit(1);
        }
    }

    switch (argc-optind) {
    case 3:
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
        K = atoi(argv[optind+2]);
        break;
    case 2:
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
        K = N;
        break;
    case 1:
        N = atoi(argv[optind]);
        K = N;
        M = N;
        break;
    default:
        break;
    }
    if (optind < argc) {
    }


    fails += test_left(M, N, K, verbose, &conf);
    fails += test_right(M, N, K, verbose, &conf);

    exit(fails);
}
