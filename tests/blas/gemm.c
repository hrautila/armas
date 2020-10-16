
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

/*
 * A = [M,K], B = [K,N] --> C = [M,N]
 *
 * test 1:
 *   a. compute C0  = A*B      : [M,K]*[K,N] = [M,N]
 *   b. compute C1  = B.T*A.T  : [N,K]*[K,M] = [N,M]
 *   c. verify  C0 == C1.T
 *
 * test 2:
 *   a. compute C0 = A*B.T     : [M,K]*[N,K]  = [M,N] if K == N
 *   b. compute C1 = B*A.T     : [K,N]*[K,M]  = [N,M] if K == N
 *   c. verify  C0 == C1.T
 *
 * test 3:
 *   a. compute C0 = A.T*B     : [K,M]*[K,N] = [M,N] if K == M
 *   b. compute C1 = B.T*A     : [N,K]*[M,K] = [N,M] if K == M
 *
 */

int test_std(int M, int N, int K, int verbose, armas_conf_t *cf)
{
    armas_dense_t A, B, C, Ct, T;
    int ok, fails = 0;
    DTYPE n0, n1;

    armas_init(&C, M, N);
    armas_init(&Ct, N, M);
    armas_init(&T, M, N);
    armas_set_values(&C, zero, 0);
    armas_set_values(&Ct, zero, 0);

    // test 1: M != N != K
    armas_init(&A, M, K);
    armas_init(&B, K, N);
    armas_set_values(&A, unitrand, 0);
    armas_set_values(&B, unitrand, 0);

    // C = A*B; C.T = B.T*A.T
    armas_mult(0.0, &C, 1.0, &A, &B, 0, cf);
    armas_mult(0.0, &Ct,1.0, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, cf);

    armas_mcopy(&T, &Ct, ARMAS_TRANS, cf);

    n0 = rel_error(&n1, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    // accept zero error
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(A, B)   == transpose(gemm(B.T, A.T))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // test 2: M != N == K
    armas_set_values(&Ct, zero, 0);
    armas_release(&A);
    armas_release(&B);
    armas_init(&A, M, N);
    armas_init(&B, N, N);
    armas_set_values(&A, unitrand, 0);
    armas_set_values(&B, unitrand, 0);
    // C = A*B.T; Ct = B*A.T
    armas_mult(0.0, &C, 1.0, &A, &B, ARMAS_TRANSB, cf);
    armas_mult(0.0, &Ct, 1.0, &B, &A, ARMAS_TRANSB, cf);
    armas_mcopy(&T, &Ct, ARMAS_TRANS, cf);

    n0 = rel_error((DTYPE *)0, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(A, B.T) == transpose(gemm(B, A.T))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // test 3: M == K != N
    armas_set_values(&Ct, zero, ARMAS_NONE);
    armas_release(&A);
    armas_release(&B);
    armas_init(&A, M, M);
    armas_init(&B, M, N);
    armas_set_values(&A, unitrand, ARMAS_NONE);
    armas_set_values(&B, unitrand, ARMAS_NONE);
    // C = A.T*B; Ct = B.T*A
    armas_mult(0.0, &C, 1.0, &A, &B, ARMAS_TRANSA, cf);
    armas_mult(0.0, &Ct,1.0, &B, &A, ARMAS_TRANSA, cf);
    armas_mcopy(&T, &Ct, ARMAS_TRANS, cf);

    n0 = rel_error(&n1, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, cf);

    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(A.T, B) == transpose(gemm(B.T, A))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // test 1: M != N != K
    armas_init(&A, M, K);
    armas_init(&B, K, N);
    armas_set_values(&A, unitrand, ARMAS_NONE);
    armas_set_values(&B, unitrand, ARMAS_NONE);

    // C = A*B; C.T = B.T*A.T
    armas_mult(0.0, &C, 1.0, &A, &B, 0, cf);
    //armas_mscale(&A, -1.0, 0, cf);
    armas_mult(0.0, &Ct, 1.0, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB/*|ARMAS_ABSB*/, cf);
    armas_mcopy(&T, &Ct, ARMAS_TRANS, cf);

    n0 = rel_error(&n1, &T, &C, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    fails += 1 - ok;
    printf("%6s: gemm(A, B)   == transpose(gemm(B.T, |-A.T|))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t conf;

    int opt;
    int N = 633;
    int M = 653;
    int K = 337;
    int verbose = 0;
    int fails = 0;

    conf = *armas_conf_default();

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

    fails += test_std(M, N, K, verbose, &conf);

    exit(fails);
}
