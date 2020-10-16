
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int upper(int M, int N, int verbose, armas_conf_t *cf)
{
    armas_dense_t C, C0, A, At, B, Bt;
    DTYPE n0, n1, alpha = 1.0;
    int ok, fails = 0;

    armas_init(&C,  M, N);
    armas_init(&C0, M, N);
    armas_init(&A,  M, N/2);
    armas_init(&B,  N/2, N);
    armas_init(&At, N/2, M);
    armas_init(&Bt, N, N/2);

    armas_set_values(&A, zeromean, ARMAS_NULL);
    armas_set_values(&B, zeromean, ARMAS_NULL);
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    armas_mcopy(&Bt, &B, ARMAS_TRANS, cf);

    printf("C(M,N) upper: M=%d, N=%d, K=%d\n", M, N, N/2);
    // upper(C)
    armas_mult(0.0, &C0, alpha, &A, &B, ARMAS_NULL, cf);
    armas_make_trm(&C0, ARMAS_UPPER);
    armas_set_values(&C, zero, ARMAS_NULL);
    armas_make_trm(&C, ARMAS_UPPER);

    // ----------------------------------------------------------------------------
    // 1. C = upper(C) + A*B
    armas_update_trm(0.0, &C, alpha, &A, &B, ARMAS_UPPER, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, U|N|N) == TriU(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // ----------------------------------------------------------------------------
    // 2. C = upper(C) + A.T*B
    armas_update_trm(0.0, &C, alpha, &At, &B, ARMAS_UPPER|ARMAS_TRANSA, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, U|T|N) == TriU(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // ----------------------------------------------------------------------------
    // 3. C = upper(C) + A*B.T
    armas_update_trm(0.0, &C, alpha, &A, &Bt, ARMAS_UPPER|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, U|N|T) == TriU(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // ----------------------------------------------------------------------------
    // 4. C = upper(C) + A.T*B.T
    armas_update_trm(0.0, &C, alpha, &At, &Bt, ARMAS_UPPER|ARMAS_TRANSA|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, U|T|T) == TriU(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int lower(int M, int N, int verbose, armas_conf_t *cf)
{
    armas_dense_t C, C0, A, At, B, Bt;
    DTYPE n0, n1, alpha = 1.0;
    int ok, fails = 0;

    armas_init(&C,  M, N);
    armas_init(&C0, M, N);
    armas_init(&A,  M, N/2);
    armas_init(&B,  N/2, N);
    armas_init(&At, N/2, M);
    armas_init(&Bt, N, N/2);

    armas_set_values(&A, zeromean, ARMAS_NULL);
    armas_set_values(&B, zeromean, ARMAS_NULL);
    armas_mcopy(&At, &A, ARMAS_TRANS, cf);
    armas_mcopy(&Bt, &B, ARMAS_TRANS, cf);

    printf("C(M,N) lower: M=%d, N=%d, K=%d\n", M, N, N/2);
    // ----------------------------------------------------------------------------
    // lower(C)
    armas_mult(0.0, &C0, alpha, &A, &B, ARMAS_NULL, cf);
    armas_make_trm(&C0, ARMAS_LOWER);
    armas_set_values(&C, zero, ARMAS_NULL);

    // ----------------------------------------------------------------------------
    // 1. C = lower(C) + A*B
    armas_update_trm(0.0, &C, alpha, &A, &B, ARMAS_LOWER, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, L|N|N) == TriL(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    // ----------------------------------------------------------------------------
    // 2. C = lower(C) + A.T*B
    armas_update_trm(0.0, &C, alpha, &At, &B, ARMAS_LOWER|ARMAS_TRANSA, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, L|T|N) == TriL(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;
    // ----------------------------------------------------------------------------
    // 3. C = lower(C) + A*B.T
    armas_update_trm(0.0, &C, alpha, &A, &Bt, ARMAS_LOWER|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, L|N|T) == TriL(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;
    // ----------------------------------------------------------------------------
    // 4. C = lower(C) + A.T*B.T
    armas_update_trm(0.0, &C, alpha, &At, &Bt, ARMAS_LOWER|ARMAS_TRANSA|ARMAS_TRANSB, cf);
    n0 = rel_error(&n1, &C, &C0, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0 || isOK(n0, N) ? 1 : 0;
    printf("%6s: trmupd(C, A, B, L|T|T) == TriL(gemm(C, A, B))\n", PASS(ok));
    if (verbose > 0) {
        printf("   || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;

    int opt;
    int N = 311, M = 353;
    int verbose = 1;
    int fails = 0;
    int all = 1;
    int do_upper = 0;
    int do_lower = 0;

    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vr:b:nUL")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'n':
            cf.optflags = ARMAS_ONAIVE;
            all = 0;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags = env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            all = 0;
            break;
        case 'b':
            env->mb = atoi(optarg);
            env->nb = env->kb = env->mb;
            cf.optflags = 0;
            all = 0;
            break;
        case 'U':
            do_upper = 1 - do_upper;
            all = 0;
            break;
        case 'L':
            do_lower = 1 - do_lower;
            all = 0;
            break;
        default:
            fprintf(stderr, "usage: testtrmk [-vr:b:n] [M N]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        M = atoi(argv[optind]);
        N = atoi(argv[optind+1]);
    } else if (optind < argc) {
        M = N = atoi(argv[optind]);
    }

    if (all) {
        fails += upper(M, N, verbose, &cf);
        fails += lower(N, M, verbose, &cf);
    } else {
        if (do_upper)
            fails += upper(M, N, verbose, &cf);
        if (do_lower)
            fails += lower(N, M, verbose, &cf);
    }

    exit(fails);
}
