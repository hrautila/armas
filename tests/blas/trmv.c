
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "testing.h"

int test_std(int N, int verbose, int unit, armas_conf_t *cf)
{
    armas_x_dense_t X, Y, X0, A, At;
    DTYPE nrm_z;
    int ok;
    int fails = 0;

    armas_x_init(&Y, N, 1);
    armas_x_init(&X0, N, 1);
    armas_x_init(&X, N, 1);
    armas_x_init(&A, N, N);
    armas_x_init(&At, N, N);

    armas_x_set_values(&X, one, ARMAS_NULL);
    armas_x_set_values(&Y, zero, ARMAS_NULL);
    armas_x_mcopy(&X0, &X, 0, cf);
    armas_x_set_values(&A, one, ARMAS_UPPER);
    if (unit) {
        for (int i = 0; i < N; i++) {
            armas_x_set(&A, i, i, 1.0);
        }
    }
    armas_x_mcopy(&At, &A, ARMAS_TRANS, cf);

    printf("** trmv: upper, %s\n", unit ? "unit-diagonal" : "");
    // --- upper ---
    armas_x_mvmult_trm(&X, -2.0, &A, ARMAS_UPPER|unit, cf);
    armas_x_mvmult(1.0, &X, 2.0, &A, &X0, ARMAS_NULL, cf);
    nrm_z = armas_x_nrm2(&X, cf);
    ok = nrm_z == 0.0 || isOK(nrm_z, N) ? 1 : 0;
    printf("%6s : trmv(X, U) == gemv(U, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", nrm_z, ndigits(nrm_z));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // --- upper, trans ---
    armas_x_mvmult_trm(&X, -2.0, &A, ARMAS_UPPER|ARMAS_TRANSA|unit, cf);
    armas_x_mvmult(1.0, &X, 2.0, &A, &X0, ARMAS_TRANSA, cf);
    nrm_z = armas_x_nrm2(&X, cf);
    ok = nrm_z == 0.0 || isOK(nrm_z, N) ? 1 : 0;
    printf("%6s : trmv(X, U^T) == gemv(U^T, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", nrm_z, ndigits(nrm_z));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    printf("** trmv: lower, %s\n", unit ? "unit-diagonal" : "");
    // --- lower ---
    armas_x_mvmult_trm(&X, -2.0, &At, ARMAS_LOWER|unit, cf);
    armas_x_mvmult(1.0, &X, 2.0, &At, &X0, ARMAS_NULL, cf);
    nrm_z = armas_x_nrm2(&X, cf);
    ok = nrm_z == 0.0 || isOK(nrm_z, N) ? 1 : 0;
    printf("%6s : trmv(X, L) == gemv(L, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", nrm_z, ndigits(nrm_z));
    }
    fails += 1 - ok;

    armas_x_mcopy(&X, &X0, 0, cf);

    // --- lower, trans ---
    armas_x_mvmult_trm(&X, -2.0, &At, ARMAS_LOWER|ARMAS_TRANSA|unit, cf);
    armas_x_mvmult(1.0, &X, 2.0, &At, &X0, ARMAS_TRANSA, cf);
    nrm_z = armas_x_nrm2(&X, cf);
    ok = nrm_z == 0.0 || isOK(nrm_z, N) ? 1 : 0;
    printf("%6s : trmv(X, L^T) == gemv(L^T, X)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", nrm_z, ndigits(nrm_z));
    }
    fails += 1 - ok;
    return fails;
}

int main(int argc, char **argv)
{
    armas_conf_t cf;
    armas_env_t *env;

    int opt;
    int N = 911;
    int fails = 0;
    int verbose = 0;
    int unit = 0;

    cf = *armas_conf_default();
    env = armas_getenv();
    while ((opt = getopt(argc, argv, "vr:ub:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'b':
            env->blas1min = atoi(optarg);
            break;
        case 'u':
            unit = ARMAS_UNIT;
            break;
        default:
            fprintf(stderr, "usage: trmv [size]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    fails += test_std(N, verbose, unit, &cf);
    fails += test_std(N, verbose, ARMAS_UNIT, &cf);

    exit(fails);
}
