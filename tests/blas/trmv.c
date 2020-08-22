
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"


int test_mvmult(int N, int verbose, int flags, armas_conf_t *cf)
{
    armas_x_dense_t X1, Y, X0, A;
    armas_env_t *env = armas_getenv();
    DTYPE n0, n1;
    int ok;
    int unit = (flags & ARMAS_UNIT) ? 1 : 0;
    const char *uplo = (flags & ARMAS_UPPER) ? "upper" : "lower";
    const char *gemv = (flags & ARMAS_TRANS) ? "A^T*x" : "A*x";
    const char *t = (flags & ARMAS_TRANS) ? "^T" : "";
    const char *u = (flags & ARMAS_UNIT) ? "unit-diag" : "non-unit";

    armas_x_init(&X0, N, 1);
    armas_x_init(&X1, N, 1);
    armas_x_init(&Y, N, 1);
    armas_x_init(&A, N, N);

    armas_x_set_values(&X0, one, 0);
    armas_x_mcopy(&Y, &X0, 0, cf);
    armas_x_set_values(&A, unitrand, flags);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, ONE);
        }
    }

    env->blas2min = 0;
    armas_x_mvmult_trm(&X0, 2.0, &A, flags, cf);
    armas_x_mvmult(ZERO, &X1, 2.0, &A, &Y, flags, cf);

    n0 = rel_error(&n1, &X0, &X1, ARMAS_NORM_TWO, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : trmv(%s(A%s), x) == %s [%s]\n", PASS(ok), uplo, t, gemv, u);
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    armas_x_release(&X0);
    armas_x_release(&X1);
    armas_x_release(&A);
    armas_x_release(&Y);
    return 1 - ok;
}

int test_blocked(int N, int verbose, int lb, int flags, armas_conf_t *cf)
{
    armas_x_dense_t X1, X0, A;
    armas_env_t *env = armas_getenv();
    DTYPE n0, n1;
    int ok;
    int unit = (flags & ARMAS_UNIT) ? 1 : 0;
    const char *uplo = (flags & ARMAS_UPPER) ? "upper" : "lower";
    const char *t = (flags & ARMAS_TRANS) ? "^T" : "";
    const char *u = (flags & ARMAS_UNIT) ? "unit-diag" : "non-unit";

    armas_x_init(&X0, N, 1);
    armas_x_init(&X1, N, 1);
    armas_x_init(&A, N, N);

    armas_x_set_values(&X0, one, 0);
    armas_x_mcopy(&X1, &X0, 0, cf);
    armas_x_set_values(&A, unitrand, flags);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, ONE);
        }
    }

    env->blas2min = 0;
    armas_x_mvmult_trm(&X0, 2.0, &A, flags, cf);
    env->blas2min = lb;
    armas_x_mvmult_trm(&X1, 2.0, &A, flags, cf);


    n0 = rel_error(&n1, &X0, &X1, ARMAS_NORM_TWO, 0, cf);
    ok = n0 == 0.0 || isOK(n0, N) ? 1 : 0;
    printf("%6s : unblk.%s(A%s)*x == blk.%s(A%s)*x [%s]\n",
         PASS(ok), uplo, t, uplo, t, u);
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }
    armas_x_release(&X0);
    armas_x_release(&X1);
    armas_x_release(&A);
    return 1 - ok;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;

    int opt;
    int N = 911;
    int fails = 0;
    int verbose = 0;
    int lb = 16;

    cf = *armas_conf_default();
    while ((opt = getopt(argc, argv, "vb:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'b':
            lb = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: trmv [size]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }
    printf("** unblocked trmv against mvmult.\n");
    fails += test_mvmult(N, verbose, ARMAS_UPPER, &cf);
    fails += test_mvmult(N, verbose, ARMAS_UPPER|ARMAS_UNIT, &cf);
    fails += test_mvmult(N, verbose, ARMAS_LOWER, &cf);
    fails += test_mvmult(N, verbose, ARMAS_LOWER|ARMAS_UNIT, &cf);

    fails += test_mvmult(N, verbose, ARMAS_UPPER|ARMAS_TRANS, &cf);
    fails += test_mvmult(N, verbose, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT, &cf);
    fails += test_mvmult(N, verbose, ARMAS_LOWER|ARMAS_TRANS, &cf);
    fails += test_mvmult(N, verbose, ARMAS_LOWER|ARMAS_TRANS|ARMAS_UNIT, &cf);

    printf("** unblocked trmv against blocked trmv.\n");

    fails += test_blocked(N, verbose, lb, ARMAS_UPPER, &cf);
    fails += test_blocked(N, verbose, lb, ARMAS_UPPER|ARMAS_UNIT, &cf);
    fails += test_blocked(N, verbose, lb, ARMAS_LOWER, &cf);
    fails += test_blocked(N, verbose, lb, ARMAS_LOWER|ARMAS_UNIT, &cf);

    fails += test_blocked(N, verbose, lb, ARMAS_UPPER|ARMAS_TRANS, &cf);
    fails += test_blocked(N, verbose, lb, ARMAS_UPPER|ARMAS_TRANS|ARMAS_UNIT, &cf);
    fails += test_blocked(N, verbose, lb, ARMAS_LOWER|ARMAS_TRANS, &cf);
    fails += test_blocked(N, verbose, lb, ARMAS_LOWER|ARMAS_TRANS|ARMAS_UNIT, &cf);

    exit(fails);
}
