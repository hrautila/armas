
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"
#define __null (armas_x_dense_t *)0

int left(int N, int K, int unit, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t X, Xg, Y, X0, A, At;
    DTYPE n0, n1;
    int ok, fails = 0;
    armas_ac_handle_t ac;

    armas_x_init(&Y, N, K);
    armas_x_init(&X0, N, K);
    armas_x_init(&X, N, K);
    armas_x_init(&Xg, N, K);
    armas_x_init(&A, N, N);
    armas_x_init(&At, N, N);
    armas_ac_init(&ac, ARMAS_AC_THREADED);
    cf->accel = ac;

    armas_x_set_values(&X, unitrand, 0);
    armas_x_set_values(&Y, zero, 0);
    armas_x_mcopy(&X0, &X, 0, cf);
    armas_x_set_values(&A, unitrand, ARMAS_UPPER);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }
    armas_x_mcopy(&At, &A, ARMAS_TRANS, cf);

    printf("** trmm: left, %s\n", unit ? "unit diagonal" : "");
    // trmm(A, X) == gemm(A, X)
    armas_x_mult_trm(&X, -2.0, &A, ARMAS_LEFT|ARMAS_UPPER|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &A, &X0, 0, cf);
    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, U|N) == gemm(upper(A), X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;
    armas_x_mcopy(&X, &X0, 0, cf);

    // trmm(A.T, X) == gemm(A.T, X)
    armas_x_mult_trm(&X, -2.0, &A,  ARMAS_LEFT|ARMAS_UPPER|ARMAS_TRANSA|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &A, &X0, ARMAS_TRANSA, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, U|T) == gemm(upper(A).T, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // trmv(A, X) == gemv(A, X)
    armas_x_mult_trm(&X, -2.0, &At,  ARMAS_LEFT|ARMAS_LOWER|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &At, &X0, 0, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, L|N) == gemm(lower(A), X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // trmv(A.T, X) == gemv(A.T, X)
    armas_x_mult_trm(&X, -2.0, &At,  ARMAS_LEFT|ARMAS_LOWER|ARMAS_TRANSA|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &At, &X0, ARMAS_TRANSA, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, L|T) == gemm(lower(A).T, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_ac_release(ac);
    armas_x_release(&Y);
    armas_x_release(&X0);
    armas_x_release(&X);
    armas_x_release(&Xg);
    armas_x_release(&A);
    armas_x_release(&At);

    return fails;
}

int right(int N, int K, int unit, int verbose, armas_conf_t *cf)
{
    armas_x_dense_t X, Y, X0, Xg, A, At;
    DTYPE n0, n1;
    int ok, fails = 0;
    armas_ac_handle_t ac;

    armas_x_init(&Y, K, N);
    armas_x_init(&X0, K, N);
    armas_x_init(&X, K, N);
    armas_x_init(&Xg, K, N);
    armas_x_init(&A, N, N);
    armas_x_init(&At, N, N);
    armas_ac_init(&ac, ARMAS_AC_THREADED);
    cf->accel = ac;

    armas_x_set_values(&X, unitrand, 0);
    armas_x_set_values(&Y, zero, 0);
    armas_x_mcopy(&X0, &X, 0, cf);
    armas_x_set_values(&A, unitrand, ARMAS_UPPER);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }
    armas_x_mcopy(&At, &A, ARMAS_TRANS, cf);

    printf("** trmm: right, %s\n", unit ? "unit diagonal" : "");
    // trmm(A, X) == gemm(A, X)
    armas_x_mult_trm(&X, -2.0, &A, ARMAS_RIGHT|ARMAS_UPPER|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &X0, &A, 0, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, U|N) == gemm(X, upper(A))\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);
    // trmm(A.T, X) == gemm(A.T, X)
    armas_x_mult_trm(&X, -2.0, &A,  ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &X0, &A, ARMAS_TRANSB, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, U|T) == gemm(X, upper(A).T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // trmv(A, X) == gemv(A, X)
    armas_x_mult_trm(&X, -2.0, &At,  ARMAS_RIGHT|ARMAS_LOWER|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &X0, &At, 0, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, L|N) == gemm(X, lower(A))\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // trmv(A.T, X) == gemv(A.T, X)
    armas_x_mult_trm(&X, -2.0, &At,  ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA|unit, cf);
    armas_x_mult(0.0, &Xg, -2.0, &X0, &At, ARMAS_TRANSB, cf);

    n0 = rel_error(&n1, &X, &Xg, ARMAS_NORM_ONE, ARMAS_NONE, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmm(X, A, L|T) == gemm(X, lower(A).T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    fails += 1 - ok;

    armas_ac_release(ac);
    armas_x_release(&Y);
    armas_x_release(&X0);
    armas_x_release(&X);
    armas_x_release(&Xg);
    armas_x_release(&A);
    armas_x_release(&At);

    return fails;
}


int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;
    int opt;
    int N = 411, K = 203;
    int fails = 0;
    int verbose = 0;
    int unit = 0;
    int all = 1;
    int do_left = 0;
    int do_right = 0;

    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vr:nb:uLR")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            all = 0;
            break;
        case 'n':
            cf.optflags |= ARMAS_ONAIVE;
            all = 0;
            break;
        case 'b':
            env->mb = atoi(optarg);
            env->nb = env->kb = env->mb;
            cf.optflags = 0;
            all = 0;
            break;
        case 'u':
            unit = ARMAS_UNIT;
            all = 0;
            break;
        case 'L':
            do_left = 1 - do_left;
            all = 0;
            break;
        case 'R':
            do_right = 1 - do_right;
            all = 0;
            break;
        default:
            fprintf(stderr, "usage: trmn [-uvLR N K]\n");
            exit(1);
        }
    }

    if (optind < argc-1) {
        N = atoi(argv[optind]);
        K = atoi(argv[optind+1]);
    } else if (optind < argc) {
        N = K = atoi(argv[optind]);
    }

    if (all) {
        fails += left(N, K, 0, verbose, &cf);
        fails += left(N, K, ARMAS_UNIT, verbose, &cf);
        fails += right(N, K, 0, verbose, &cf);
        fails += right(N, K, ARMAS_UNIT, verbose, &cf);
    } else {
        if (do_left)
            fails += left(N, K, unit, verbose, &cf);
        if (do_right)
            fails += right(N, K, unit, verbose, &cf);
    }

    exit(fails);
}
