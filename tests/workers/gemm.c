
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

/*
 * A = [M,K], B = [K,N] --> C = [M,N]
 *
 */
int test_std(int M, int N, int K, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t A, B, C, Ct, T;
    int ok, fails = 0;
    DTYPE n0, n1;
    armas_ac_handle_t ac;
    armas_ac_init(&ac, ARMAS_AC_WORKERS);

    armas_x_init(&C, M, N);
    armas_x_init(&Ct, N, M);
    armas_x_init(&T, M, N);
    armas_x_set_values(&C, zero, 0);
    armas_x_set_values(&Ct, zero, 0);

    armas_x_init(&A, M, K);
    armas_x_init(&B, K, N);
    armas_x_set_values(&A, unitrand, 0);
    armas_x_set_values(&B, unitrand, 0);

    // C = A*B; C.T = B.T*A.T
    cf->accel = ac;
    armas_x_mult(ZERO, &C, ONE, &A, &B, 0, cf);
    cf->accel = (armas_ac_handle_t *)0;
    armas_x_mult(ZERO, &Ct, ONE, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &Ct, &C, ARMAS_NORM_ONE, ARMAS_TRANS, cf);

    // accept zero error
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(A, B)[Ncpu]     == transpose(gemm(B.T, A.T))[1cpu]\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    cf->accel = ac;
    armas_x_mult(ZERO, &Ct, ONE, &B, &A, ARMAS_TRANSA|ARMAS_TRANSB, cf);
    cf->accel = (struct armas_accel *)0;
    armas_x_mult(ZERO, &C, ONE, &A, &B, 0, cf);

    n0 = rel_error(&n1, &Ct, &C, ARMAS_NORM_ONE, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(B.T, A.T)]Ncpu] == transpose(gemm(A, B))[1cpu]\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // B = B.T
    armas_x_release(&B);
    armas_x_init(&B, N, K);
    armas_x_set_values(&B, unitrand, 0);

    // C = A*B.T
    cf->accel = ac;
    armas_x_mult(ZERO, &C, ONE, &A, &B, ARMAS_TRANSB, cf);
    cf->accel = (struct armas_accel *)0;
    armas_x_mult(ZERO, &Ct, ONE, &B, &A, ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &Ct, &C, ARMAS_NORM_ONE, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(A, B.T)[Ncpu]   == transpose(gemm(B, A.T))[1cpu]\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // A = A.T
    armas_x_release(&B);
    armas_x_release(&A);
    armas_x_init(&A, K, M);
    armas_x_init(&B, K, N);
    armas_x_set_values(&A, unitrand, 0);
    armas_x_set_values(&B, unitrand, 0);

    // C = A.T*B
    cf->accel = ac;
    armas_x_mult(ZERO, &C, ONE, &A, &B, ARMAS_TRANSA, cf);
    cf->accel = (struct armas_accel *)0;
    armas_x_mult(ZERO, &Ct, ONE, &B, &A, ARMAS_TRANSA, cf);
    n0 = rel_error(&n1, &Ct, &C, ARMAS_NORM_ONE, ARMAS_TRANS, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: gemm(A.T, B)[Ncpu]   == transpose(gemm(B.T, A)[1cpu])\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_ac_release(ac);
    armas_x_release(&A);
    armas_x_release(&B);
    armas_x_release(&C);
    armas_x_release(&Ct);
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

    printf("argc: %d, optind: %d\n", argc, optind);
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


    fails += test_std(M, N, K, verbose, &conf);

    exit(fails);
}
