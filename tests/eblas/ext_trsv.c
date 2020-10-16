
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "testing.h"


int test_ext(int N, int verbose, int unit, armas_conf_t *cf)
{
    armas_dense_t Y0, Y, A, At, L, Lt, E, Et;
    DTYPE n0;
    int ok;
    int fails = 0;

    armas_init(&Y, N, 1);
    armas_init(&Y0, N, 1);
    armas_init(&E, N, 1);
    armas_init(&Et, N, 1);
    armas_init(&A, N, N);
    armas_init(&At, N, N);
    armas_init(&L, N, N);
    armas_init(&Lt, N, N);

    // expect: [0, N-1*1]
    armas_set_values(&Y, one, ARMAS_NULL);
    make_ext_trsv_data(N, 0, &A, &At, &E, &Et);
    armas_mcopy(&Y0, &E, 0, cf);

    armas_mcopy(&Lt, &A, ARMAS_TRANS, cf);
    armas_mcopy(&L, &At, ARMAS_TRANS, cf);

    if (verbose > 2) {
        armas_dense_t row;
        MAT_PRINT("A", &A);
        MAT_PRINT("E", armas_col_as_row(&row, &E));
        MAT_PRINT("A^T", &At);
        MAT_PRINT("E^T", armas_col_as_row(&row, &Et));
    }

    // -------------------------------------------------------------
    // compute ||A^*Y - 1||

    armas_ext_mvsolve_trm(&Y0, 2.0, &A, ARMAS_UPPER, cf);
    armas_madd(&Y0, -2.0, 0, cf);
    n0 = armas_nrm2(&Y0, cf) / N;

    ok = !isNAN(n0) && (n0 == 0.0 || isOK(n0, N));
    fails += (1 - ok);
    printf("%6s : expected == ext_trsv(X, U)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // -------------------------------------------------------------
    armas_mcopy(&Y0, &Et, 0, cf);
    armas_ext_mvsolve_trm(&Y0, -2.0, &At, ARMAS_UPPER | ARMAS_TRANS, cf);
    armas_madd(&Y0, 2.0, 0, cf);
    n0 = armas_nrm2(&Y0, cf) / N;

    ok = !isNAN(n0) && (n0 == 0.0 || isOK(n0, N));
    fails += (1 - ok);
    printf("%6s : expected == ext_trsv(X, U|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // -------------------------------------------------------------
    // compute ||A^*Y - 1||
    armas_mcopy(&Y0, &Et, 0, cf);
    armas_ext_mvsolve_trm(&Y0, 0.5, &L, ARMAS_LOWER, cf);
    armas_madd(&Y0, -0.5, 0, cf);
    n0 = armas_nrm2(&Y0, cf) / N;

    ok = !isNAN(n0) && (n0 == 0.0 || isOK(n0, N));
    fails += (1 - ok);
    printf("%6s : expected == ext_trsv(X, L)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    // -------------------------------------------------------------
    // compute sum(A^*Y - 1)
    armas_mcopy(&Y0, &E, 0, cf);
    armas_ext_mvsolve_trm(&Y0, -0.5, &Lt, ARMAS_LOWER | ARMAS_TRANS, cf);
    armas_madd(&Y0, 0.5, 0, cf);
    n0 = armas_nrm2(&Y0, cf) / N;

    ok = !isNAN(n0) && (n0 == 0.0 || isOK(n0, N));
    fails += (1 - ok);
    printf("%6s : expected == ext_trsv(X, L|T)\n", PASS(ok));
    if (verbose > 0) {
        printf("    || rel error || : %e, [%d]\n", n0, ndigits(n0));
    }

    armas_release(&Y);
    armas_release(&Y0);
    armas_release(&E);
    armas_release(&Et);
    armas_release(&A);
    armas_release(&At);
    armas_release(&L);
    armas_release(&Lt);

    return fails;
}

int main(int argc, char **argv)
{
    int opt;
    int N = 77;
    int verbose = 0;
    int unit = 0;
    armas_env_t *env = armas_getenv();
    armas_conf_t cf = *armas_conf_default();

    while ((opt = getopt(argc, argv, "vr:u")) != -1) {
        switch (opt) {
        case 'v':
            verbose++;
            break;
        case 'r':
            env->blas2min = atoi(optarg);
            cf.optflags |= env->blas2min != 0 ? ARMAS_ORECURSIVE : ARMAS_ONAIVE;
            break;
        case 'u':
            unit = ARMAS_UNIT;
            break;
        default:
            fprintf(stderr, "usage: trsv [size]\n");
            exit(1);
        }
    }

    if (optind < argc) {
        N = atoi(argv[optind]);
    }

    int fails = 0;

    fails += test_ext(N, verbose, unit, &cf);
    exit(fails);
}
